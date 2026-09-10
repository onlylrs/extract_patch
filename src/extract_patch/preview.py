from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

from .config import AppConfig
from .concurrency import process_map, raise_if_system_error
from .filters import PatchFilterPipeline
from .heuristics import HeuristicPipeline
from .models import RunResult, SlideResult, SlideSpec
from .planner import effective_patching, plan_patches
from .readers import open_reader
from .reporting import RunLogger, make_run_id, save_center_previews, save_overlay
from .sinks.base import prepare_image


def _preview_slide(
    spec: SlideSpec,
    config: AppConfig,
    output_root: Path,
) -> SlideResult:
    started = time.perf_counter()
    cv2.setNumThreads(config.parallel.opencv_threads)
    destination = output_root / spec.slide_id
    destination.mkdir(parents=True, exist_ok=True)
    try:
        with open_reader(spec, config.reader) as reader:
            metadata = reader.metadata
            width = config.reader.thumbnail_width
            height = max(1, round(width * metadata.dimensions[1] / metadata.dimensions[0]))
            thumbnail = reader.thumbnail((width, height)).convert("RGB")
            decision = HeuristicPipeline.from_config(config).run(
                np.asarray(thumbnail), metadata.dimensions
            )
            if decision.status != "accepted" or decision.region is None:
                raise RuntimeError(f"Heuristic pipeline failed: {decision.reason}")
            effective = effective_patching(
                config.patching,
                metadata.mpp,
                metadata.level_downsamples[config.patching.level],
            )
            plans = plan_patches(metadata, decision.region, effective)
            rng = np.random.default_rng(config.preview.seed)
            n = min(config.preview.n_patches, len(plans))
            candidate_indices = rng.permutation(len(plans)).tolist() if n else []
            save_overlay(thumbnail, decision, plans, destination / "contour.jpg")
            sample_dir = destination / "patches"
            if candidate_indices:
                sample_dir.mkdir(parents=True, exist_ok=True)
            filter_pipeline = PatchFilterPipeline.from_config(config)
            saved = 0
            for index in candidate_indices:
                plan = plans[index]
                patch = reader.read_region(
                    (plan.x, plan.y),
                    plan.level,
                    (plan.read_size, plan.read_size),
                ).convert("RGB")
                patch = prepare_image(patch, plan)
                if not filter_pipeline.run(patch, plan).keep:
                    continue
                patch.save(
                    sample_dir
                    / f"sample_{saved:02d}_x{plan.x}_y{plan.y}_l{plan.level}.jpg",
                    quality=92,
                )
                saved += 1
                if saved >= n:
                    break
        result = SlideResult(
            slide_id=spec.slide_id,
            status="success",
            patch_count=len(plans),
            reader=metadata.reader,
            heuristic=decision.name,
            color_correction=metadata.properties.get("extract_patch.color_correction"),
            elapsed_seconds=time.perf_counter() - started,
        )
    except Exception as exc:
        raise_if_system_error(exc)
        result = SlideResult(
            slide_id=spec.slide_id,
            status="failed",
            elapsed_seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
    return result


def _center_preview_slide(
    spec: SlideSpec,
    config: AppConfig,
    output_root: Path,
) -> SlideResult:
    started = time.perf_counter()
    cv2.setNumThreads(config.parallel.opencv_threads)
    try:
        with open_reader(spec, config.reader) as reader:
            metadata = reader.metadata
            width = config.reader.thumbnail_width
            height = max(1, round(width * metadata.dimensions[1] / metadata.dimensions[0]))
            thumbnail = reader.thumbnail((width, height)).convert("RGB")
        decision = HeuristicPipeline.from_config(config).run(
            np.asarray(thumbnail), metadata.dimensions
        )
        if decision.status != "accepted" or decision.region is None:
            raise RuntimeError(f"Heuristic pipeline failed: {decision.reason}")
        effective = effective_patching(
            config.patching,
            metadata.mpp,
            metadata.level_downsamples[config.patching.level],
        )
        plans = plan_patches(metadata, decision.region, effective)
        save_center_previews(thumbnail, decision, plans, output_root, spec.slide_id)
        result = SlideResult(
            slide_id=spec.slide_id,
            status="success",
            reader=metadata.reader,
            heuristic=decision.name,
            color_correction=metadata.properties.get("extract_patch.color_correction"),
            elapsed_seconds=time.perf_counter() - started,
        )
    except Exception as exc:
        raise_if_system_error(exc)
        result = SlideResult(
            slide_id=spec.slide_id,
            status="failed",
            elapsed_seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
    return result


def _run_preview_workers(
    worker,
    specs: list[SlideSpec],
    config: AppConfig,
    output_root: Path,
    logger: RunLogger,
    process_name: str,
) -> tuple[list[SlideResult], dict]:
    try:
        results = process_map(
            worker,
            specs,
            config,
            output_root,
            max_workers=config.parallel.slide_workers,
            process_name=process_name,
            on_result=logger.record_slide,
        )
        return results, logger.finalize(results)
    except BaseException:
        logger.abort("System-level preview failure; all WSI workers were stopped")
        raise


def preview(
    specs: Iterable[SlideSpec],
    config: AppConfig,
    *,
    run_id: str | None = None,
) -> RunResult:
    specs = list(specs)
    run_id = run_id or make_run_id("preview")
    output_root = Path(config.preview.root) / run_id
    logger = RunLogger(Path(config.logging.root), run_id, config)
    logger.logger.info("Starting preview for %d slides", len(specs))
    results, summary = _run_preview_workers(
        _preview_slide, specs, config, output_root, logger, "preview-wsi"
    )
    return RunResult(
        run_id=run_id,
        status="success" if summary["status"] == "success" else "partial",
        slides=results,
        output_root=output_root,
        log_dir=logger.path,
    )


def center_preview(
    specs: Iterable[SlideSpec],
    config: AppConfig,
    *,
    run_id: str | None = None,
) -> RunResult:
    specs = list(specs)
    run_id = run_id or make_run_id("center_preview")
    output_root = Path(config.output.root).parent / "preview"
    logger = RunLogger(Path(config.logging.root), run_id, config)
    logger.logger.info("Starting center preview for %d slides", len(specs))
    results, summary = _run_preview_workers(
        _center_preview_slide, specs, config, output_root, logger, "center-preview-wsi"
    )
    return RunResult(
        run_id=run_id,
        status="success" if summary["status"] == "success" else "partial",
        slides=results,
        output_root=output_root,
        log_dir=logger.path,
    )
