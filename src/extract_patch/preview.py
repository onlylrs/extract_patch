from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

from .config import AppConfig, parse_permission_mode
from .concurrency import process_map, raise_if_system_error
from .filters import PatchFilterPipeline
from .filters.qc.runtime import bind_qc_client
from .filters.qc.service import running_qc_service
from .heuristics import HeuristicPipeline
from .models import RunResult, SlideResult, SlideSpec
from .planner import effective_patching, plan_patches
from .permissions import chmod_tree
from .readers import open_reader
from .reporting import RunLogger, make_run_id, save_center_previews, save_overlay
from .sinks.base import prepare_image


def _preview_slide(
    spec: SlideSpec,
    config: AppConfig,
    output_root: Path,
    qc_client=None,
) -> SlideResult:
    bind_qc_client(qc_client)
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
    specs: Iterable[SlideSpec],
    config: AppConfig,
    output_root: Path,
    logger: RunLogger,
    process_name: str,
    qc_client=None,
) -> tuple[list[SlideResult], dict]:
    resolved = 0

    def iter_logged_specs():
        nonlocal resolved
        for spec in specs:
            resolved += 1
            if resolved == 1 or resolved % 100 == 0:
                logger.logger.info(
                    "input_progress resolved=%d latest_slide=%s",
                    resolved,
                    spec.slide_id,
                )
            yield spec

    try:
        results = process_map(
            worker,
            iter_logged_specs(),
            config,
            output_root,
            qc_client,
            max_workers=config.parallel.slide_workers,
            process_name=process_name,
            on_result=logger.record_slide,
        )
        logger.logger.info("Input stream complete: resolved=%d", resolved)
        chmod_tree(output_root, parse_permission_mode(config.output.permissions))
        return results, logger.finalize(results)
    except BaseException:
        try:
            chmod_tree(output_root, parse_permission_mode(config.output.permissions))
        except OSError:
            logger.logger.exception("Could not apply permissions to partial preview outputs")
        logger.abort("System-level preview failure; all WSI workers were stopped")
        raise


def preview(
    specs: Iterable[SlideSpec],
    config: AppConfig,
    *,
    run_id: str | None = None,
) -> RunResult:
    run_id = run_id or make_run_id("preview")
    output_root = Path(config.preview.root) / run_id
    logger = RunLogger(Path(config.logging.root), run_id, config)
    logger.logger.info("Starting preview; inputs will be resolved incrementally")
    with running_qc_service(config, logger.logger) as qc_client:
        results, summary = _run_preview_workers(
            _preview_slide, specs, config, output_root, logger, "preview-wsi", qc_client
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
    run_id = run_id or make_run_id("center_preview")
    output_root = Path(config.output.root).parent / "preview"
    logger = RunLogger(Path(config.logging.root), run_id, config)
    logger.logger.info(
        "Starting center preview; inputs will be resolved incrementally "
        "and completed previews will be skipped"
    )
    resolved = 0
    ordered_ids: list[str] = []
    results_by_id: dict[str, SlideResult] = {}

    def preview_is_complete(slide_id: str) -> bool:
        destinations = (
            output_root / "thumbnail" / f"{slide_id}.jpeg",
            output_root / "mask" / f"{slide_id}.jpg",
        )
        return all(path.is_file() and path.stat().st_size > 0 for path in destinations)

    def iter_pending_specs():
        nonlocal resolved
        for spec in specs:
            resolved += 1
            ordered_ids.append(spec.slide_id)
            if resolved == 1 or resolved % 100 == 0:
                logger.logger.info(
                    "input_progress resolved=%d latest_slide=%s",
                    resolved,
                    spec.slide_id,
                )
            if not config.output.overwrite and preview_is_complete(spec.slide_id):
                result = SlideResult(spec.slide_id, "skipped")
                results_by_id[spec.slide_id] = result
                logger.record_slide(result)
                continue
            yield spec

    def record_result(result: SlideResult) -> None:
        results_by_id[result.slide_id] = result
        logger.record_slide(result)

    try:
        process_map(
            _center_preview_slide,
            iter_pending_specs(),
            config,
            output_root,
            max_workers=config.parallel.slide_workers,
            process_name="center-preview-wsi",
            on_result=record_result,
        )
        logger.logger.info("Input stream complete: resolved=%d", resolved)
        results = [results_by_id[slide_id] for slide_id in ordered_ids]
        chmod_tree(output_root, parse_permission_mode(config.output.permissions))
        summary = logger.finalize(results)
    except BaseException:
        try:
            chmod_tree(output_root, parse_permission_mode(config.output.permissions))
        except OSError:
            logger.logger.exception("Could not apply permissions to partial center previews")
        logger.abort("System-level center preview failure; all WSI workers were stopped")
        raise
    return RunResult(
        run_id=run_id,
        status="success" if summary["status"] == "success" else "partial",
        slides=results,
        output_root=output_root,
        log_dir=logger.path,
    )
