from __future__ import annotations

import dataclasses
import json
import os
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import ExitStack, nullcontext
from pathlib import Path
from queue import Queue
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

from .config import AppConfig
from .filters import FilterDecision, PatchFilterPipeline
from .heuristics import HeuristicPipeline
from .models import PatchPlan, PatchRecord, RunResult, SlideResult, SlideSpec
from .planner import effective_patching, plan_patches
from .readers import open_reader, stage_slide
from .reporting import RunLogger, make_run_id
from .sinks import PatchSink, create_sink, patch_stem
from .sinks.base import EncodedPatch, prepare_image


def _expected_name(sink: PatchSink, plan: PatchPlan) -> str:
    extension = getattr(sink, "extension", "")
    return patch_stem(plan) + extension


def _read_patch(reader, plan: PatchPlan) -> Image.Image:
    image = reader.read_region((plan.x, plan.y), plan.level, (plan.read_size, plan.read_size))
    return image.convert("RGB")


def _write_patch_index(
    output_dir: Path,
    outputs_by_plan_index: dict[int, str],
    *,
    tar_mode: bool,
) -> None:
    patches = []
    for index, (_plan_index, output) in enumerate(sorted(outputs_by_plan_index.items())):
        if tar_mode:
            shard, name = output.split("/", 1)
            patches.append({"index": index, "name": name, "shard": shard})
        else:
            patches.append({"index": index, "name": output})

    payload = {"patch_count": len(patches), "patches": patches}
    fd, temporary_name = tempfile.mkstemp(
        prefix=".index-",
        suffix=".json.tmp",
        dir=output_dir,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_dir / "index.json")
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _extract_batches(
    spec: SlideSpec,
    active_spec: SlideSpec,
    plans: list[PatchPlan],
    config: AppConfig,
    sink: PatchSink,
    logger: RunLogger,
    reader_config,
) -> tuple[int, int, list[str], dict[str, float], dict[int, str]]:
    worker_count = max(1, min(config.parallel.read_workers_per_slide, len(plans) or 1))
    estimated_bytes = max(1, config.patching.output_size**2 * 3)
    byte_limited = max(1, config.parallel.max_inflight_bytes // estimated_bytes)
    batch_size = max(1, min(config.parallel.max_inflight_patches, byte_limited))
    errors: list[str] = []
    success = 0
    filtered = 0
    outputs_by_plan_index: dict[int, str] = {}
    timings = {"read": 0.0, "encode": 0.0, "write": 0.0}
    timings["max_inflight_patches"] = float(batch_size)
    completed = logger.completed_outputs(spec.slide_id)
    filter_pipeline = PatchFilterPipeline.from_config(config)

    with ExitStack() as stack:
        readers = [
            stack.enter_context(open_reader(active_spec, reader_config)) for _ in range(worker_count)
        ]
        pool: Queue = Queue()
        for reader in readers:
            pool.put(reader)

        def read_with_pool(plan: PatchPlan) -> Image.Image:
            reader = pool.get()
            try:
                return _read_patch(reader, plan)
            finally:
                pool.put(reader)

        def filter_and_encode(
            image: Image.Image,
            plan: PatchPlan,
        ) -> tuple[FilterDecision, EncodedPatch | None]:
            prepared = prepare_image(image, plan)
            filter_result = filter_pipeline.run(prepared, plan)
            if not filter_result.keep:
                return filter_result, None
            return filter_result, sink.encode(prepared, plan)

        with (
            ThreadPoolExecutor(max_workers=worker_count) as read_executor,
            ThreadPoolExecutor(max_workers=config.parallel.encode_workers) as encode_executor,
            ThreadPoolExecutor(max_workers=config.parallel.writer_workers) as write_executor,
        ):
            for start in range(0, len(plans), batch_size):
                batch = plans[start : start + batch_size]
                pending_reads: dict[Future, PatchPlan] = {}
                for plan in batch:
                    expected = _expected_name(sink, plan)
                    relative = f"{spec.slide_id}/{expected}" if expected else ""
                    if relative and relative in completed:
                        logger.record_patch(
                            spec.slide_id,
                            PatchRecord(
                                plan.index,
                                plan.x,
                                plan.y,
                                plan.level,
                                plan.output_size,
                                relative,
                                "skipped",
                            ),
                        )
                        outputs_by_plan_index[plan.index] = expected
                        success += 1
                        continue
                    pending_reads[read_executor.submit(read_with_pool, plan)] = plan

                read_started = time.perf_counter()
                pending_encodes: dict[Future, PatchPlan] = {}
                for future in as_completed(pending_reads):
                    plan = pending_reads[future]
                    try:
                        image = future.result()
                        pending_encodes[
                            encode_executor.submit(filter_and_encode, image, plan)
                        ] = plan
                    except Exception as exc:
                        message = f"read {plan.x},{plan.y}: {type(exc).__name__}: {exc}"
                        errors.append(message)
                        logger.record_patch(
                            spec.slide_id,
                            PatchRecord(
                                plan.index,
                                plan.x,
                                plan.y,
                                plan.level,
                                plan.output_size,
                                "",
                                "failed",
                                message,
                            ),
                        )
                timings["read"] += time.perf_counter() - read_started

                encode_started = time.perf_counter()
                pending_writes: dict[Future, tuple[PatchPlan, str]] = {}
                for future in as_completed(pending_encodes):
                    plan = pending_encodes[future]
                    try:
                        filter_result, encoded = future.result()
                        if not filter_result.keep:
                            logger.record_patch(
                                spec.slide_id,
                                PatchRecord(
                                    plan.index,
                                    plan.x,
                                    plan.y,
                                    plan.level,
                                    plan.output_size,
                                    "",
                                    "filtered",
                                    filter_result.reason,
                                ),
                            )
                            filtered += 1
                            continue
                        if encoded is None:
                            raise RuntimeError("Accepted patch filter returned no encoded patch")
                        pending_writes[write_executor.submit(sink.publish, encoded)] = (
                            plan,
                            encoded.name,
                        )
                    except Exception as exc:
                        message = f"encode {plan.x},{plan.y}: {type(exc).__name__}: {exc}"
                        errors.append(message)
                        logger.record_patch(
                            spec.slide_id,
                            PatchRecord(
                                plan.index,
                                plan.x,
                                plan.y,
                                plan.level,
                                plan.output_size,
                                "",
                                "failed",
                                message,
                            ),
                        )
                timings["encode"] += time.perf_counter() - encode_started

                write_started = time.perf_counter()
                for future in as_completed(pending_writes):
                    plan, encoded_name = pending_writes[future]
                    try:
                        published = future.result()
                        output = published or encoded_name
                        relative = f"{spec.slide_id}/{output}" if output else ""
                        logger.record_patch(
                            spec.slide_id,
                            PatchRecord(
                                plan.index,
                                plan.x,
                                plan.y,
                                plan.level,
                                plan.output_size,
                                relative,
                                "success",
                            ),
                        )
                        outputs_by_plan_index[plan.index] = output
                        success += 1
                    except FileExistsError as exc:
                        target = sink.output_dir / encoded_name
                        if target.is_file() and target.stat().st_size > 0:
                            logger.record_patch(
                                spec.slide_id,
                                PatchRecord(
                                    plan.index,
                                    plan.x,
                                    plan.y,
                                    plan.level,
                                    plan.output_size,
                                    f"{spec.slide_id}/{encoded_name}",
                                    "skipped",
                                ),
                            )
                            outputs_by_plan_index[plan.index] = encoded_name
                            success += 1
                            continue
                        message = f"write {plan.x},{plan.y}: {exc}"
                        errors.append(message)
                    except Exception as exc:
                        message = f"write {plan.x},{plan.y}: {type(exc).__name__}: {exc}"
                        errors.append(message)
                        logger.record_patch(
                            spec.slide_id,
                            PatchRecord(
                                plan.index,
                                plan.x,
                                plan.y,
                                plan.level,
                                plan.output_size,
                                "",
                                "failed",
                                message,
                            ),
                        )
                timings["write"] += time.perf_counter() - write_started
    timings["filtered_patches"] = float(filtered)
    return success, filtered, errors, timings, outputs_by_plan_index


def _extract_slide(spec: SlideSpec, config: AppConfig, logger: RunLogger) -> SlideResult:
    started = time.perf_counter()
    cv2.setNumThreads(config.parallel.opencv_threads)
    staging = (
        stage_slide(spec, config.reader.staging_root, cleanup=config.reader.staging_cleanup)
        if config.reader.staging_enabled
        else nullcontext(spec)
    )
    try:
        with staging as active_spec:
            reader_config = dataclasses.replace(config.reader, staging_enabled=False)
            segment_started = time.perf_counter()
            with open_reader(active_spec, reader_config) as reader:
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
            if not plans:
                raise RuntimeError("No patches were planned for the accepted region")
            segment_time = time.perf_counter() - segment_started

            slide_output = Path(config.output.root) / spec.slide_id
            sink = create_sink(config.output, slide_output)
            with sink:
                count, filtered, errors, timings, outputs_by_plan_index = _extract_batches(
                    spec,
                    active_spec,
                    plans,
                    config,
                    sink,
                    logger,
                    reader_config,
                )
            if errors or count + filtered != len(plans):
                raise RuntimeError(
                    f"{len(errors)} patch errors; completed {count}+{filtered} filtered/"
                    f"{len(plans)}"
                    + (f"; first: {errors[0]}" if errors else "")
                )
            if config.output.mode != "none":
                _write_patch_index(
                    slide_output,
                    outputs_by_plan_index,
                    tar_mode=config.output.mode == "tar",
                )
            elapsed = time.perf_counter() - started
            result = SlideResult(
                slide_id=spec.slide_id,
                status="success",
                patch_count=count,
                reader=metadata.reader,
                heuristic=decision.name,
                elapsed_seconds=elapsed,
                timings={
                    "segmentation": segment_time,
                    **timings,
                    "patches_per_second": count / max(elapsed, 1e-9),
                },
            )
    except Exception as exc:
        result = SlideResult(
            slide_id=spec.slide_id,
            status="failed",
            elapsed_seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
    logger.record_slide(result)
    return result


def extract(
    specs: Iterable[SlideSpec],
    config: AppConfig,
    *,
    run_id: str | None = None,
) -> RunResult:
    specs = list(specs)
    run_id = run_id or make_run_id("extract")
    output_root = Path(config.output.root)
    output_root.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(Path(config.logging.root), run_id, config)
    logger.logger.info("Starting extraction for %d slides", len(specs))
    results_by_id: dict[str, SlideResult] = {}
    with ThreadPoolExecutor(max_workers=config.parallel.slide_workers) as executor:
        futures = {executor.submit(_extract_slide, spec, config, logger): spec for spec in specs}
        for future in as_completed(futures):
            result = future.result()
            results_by_id[result.slide_id] = result
    results = [results_by_id[spec.slide_id] for spec in specs]
    summary = logger.finalize(results)
    return RunResult(
        run_id=run_id,
        status="success" if summary["status"] == "success" else "partial",
        slides=results,
        output_root=output_root,
        log_dir=logger.path,
    )
