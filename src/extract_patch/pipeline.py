from __future__ import annotations

import dataclasses
import json
import os
import tarfile
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

from .config import AppConfig, parse_permission_mode
from .concurrency import process_map, raise_if_system_error
from .filters import PatchFilterPipeline
from .filters.qc.runtime import bind_qc_client
from .filters.qc.service import running_qc_service
from .heuristics import HeuristicPipeline
from .models import PatchPlan, PatchRecord, RunResult, SlideResult, SlideSpec, SlideWorkResult
from .planner import effective_patching, plan_patches
from .permissions import chmod_tree, chmod_trees
from .readers import open_reader, stage_slide
from .reporting import RunLogger, make_run_id, save_center_previews
from .sinks import PatchSink, create_sink, generate_tar_preview, patch_stem
from .sinks.base import prepare_image


class _PatchCollector:
    """Collect one WSI's manifest entries inside its worker process."""

    def __init__(self, completed: dict[str, str]) -> None:
        self.completed = completed
        self.records: list[PatchRecord] = []

    def completed_outputs(self, _slide_id: str) -> dict[str, str]:
        return self.completed

    def record_patch(self, _slide_id: str, record: PatchRecord) -> None:
        self.records.append(record)


def _chmod_extract_outputs(config: AppConfig, output_root: Path) -> None:
    mode = parse_permission_mode(config.output.permissions)
    chmod_tree(output_root, mode)
    if config.output.mode == "none":
        return
    preview_root = output_root.parent / "preview"
    chmod_trees((preview_root / "thumbnail", preview_root / "mask"), mode)
    if preview_root.exists():
        preview_root.chmod(mode)


def _expected_name(sink: PatchSink, plan: PatchPlan) -> str:
    extension = getattr(sink, "extension", "")
    return patch_stem(plan) + extension


def _read_patch(reader, plan: PatchPlan) -> Image.Image:
    image = reader.read_region((plan.x, plan.y), plan.level, (plan.read_size, plan.read_size))
    return image.convert("RGB")


def _discover_completed_outputs(output_dir: Path, mode: str) -> dict[str, str]:
    if not output_dir.is_dir():
        return {}
    if mode in {"jpeg", "png"}:
        extension = ".jpeg" if mode == "jpeg" else ".png"
        return {
            path.name: path.name
            for path in output_dir.glob(f"*{extension}")
            if path.is_file() and path.stat().st_size > 0
        }
    if mode != "tar":
        return {}
    completed: dict[str, str] = {}
    for shard in sorted(output_dir.glob(f"{output_dir.name}_*.tar")):
        if not shard.is_file() or shard.stat().st_size <= 0:
            continue
        with tarfile.open(shard, mode="r") as archive:
            for member in archive:
                if member.isfile() and member.name.lower().endswith((".jpg", ".jpeg")):
                    completed.setdefault(member.name, f"{shard.name}/{member.name}")
    return completed


def _complete_index_patch_count(output_dir: Path, mode: str) -> int | None:
    index_path = output_dir / "index.json"
    if mode == "none" or not index_path.is_file():
        return None
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        patches = payload["patches"]
        patch_count = int(payload["patch_count"])
        if patch_count != len(patches):
            return None
        for patch in patches:
            name = patch["name"]
            target = output_dir / patch["shard"] if mode == "tar" else output_dir / name
            if not target.is_file() or target.stat().st_size <= 0:
                return None
        return patch_count
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return None


def _preview_is_complete(config: AppConfig, slide_id: str, patch_count: int) -> bool:
    preview_root = Path(config.output.root).parent / "preview"
    thumbnail = preview_root / "thumbnail" / f"{slide_id}.jpeg"
    mask = preview_root / "mask" / f"{slide_id}.jpg"
    if not all(path.is_file() and path.stat().st_size > 0 for path in (thumbnail, mask)):
        return False
    if config.output.mode == "tar" and config.output.tar_preview:
        sample_dir = Path(config.output.root) / slide_id / "sample"
        expected = min(config.output.tar_preview_n, patch_count)
        actual = sum(1 for path in sample_dir.glob("*.jpeg") if path.is_file())
        return actual >= expected
    return True


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
    logger: _PatchCollector,
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
    timings = {"read": 0.0, "filter": 0.0, "encode": 0.0, "write": 0.0}
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
                    completed_output = completed.get(expected)
                    if completed_output is not None:
                        relative = f"{spec.slide_id}/{completed_output}"
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
                        outputs_by_plan_index[plan.index] = completed_output
                        success += 1
                        continue
                    pending_reads[read_executor.submit(read_with_pool, plan)] = plan

                read_started = time.perf_counter()
                prepared_items: list[tuple[Image.Image, PatchPlan]] = []
                for future in as_completed(pending_reads):
                    plan = pending_reads[future]
                    try:
                        prepared_items.append((prepare_image(future.result(), plan), plan))
                    except Exception as exc:
                        raise_if_system_error(exc)
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

                filter_started = time.perf_counter()
                decisions = filter_pipeline.run_many(prepared_items)
                timings["filter"] += time.perf_counter() - filter_started

                encode_started = time.perf_counter()
                pending_encodes: dict[Future, PatchPlan] = {}
                for (image, plan), filter_result in zip(prepared_items, decisions):
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
                    pending_encodes[encode_executor.submit(sink.encode, image, plan)] = plan

                pending_writes: dict[Future, tuple[PatchPlan, str]] = {}
                for future in as_completed(pending_encodes):
                    plan = pending_encodes[future]
                    try:
                        encoded = future.result()
                        pending_writes[write_executor.submit(sink.publish, encoded)] = (
                            plan,
                            encoded.name,
                        )
                    except Exception as exc:
                        raise_if_system_error(exc)
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
                        raise_if_system_error(exc)
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


def _extract_slide(
    item: tuple[SlideSpec, dict[str, str]],
    config: AppConfig,
    qc_client=None,
) -> SlideWorkResult:
    bind_qc_client(qc_client)
    spec, completed = item
    collector = _PatchCollector(completed)
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
                    collector,
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
                if config.output.mode == "tar" and config.output.tar_preview:
                    generate_tar_preview(
                        slide_output,
                        count=config.output.tar_preview_n,
                        seed=config.output.tar_preview_seed,
                    )
                save_center_previews(
                    thumbnail,
                    decision,
                    plans,
                    Path(config.output.root).parent / "preview",
                    spec.slide_id,
                )
            elapsed = time.perf_counter() - started
            result = SlideResult(
                slide_id=spec.slide_id,
                status="success",
                patch_count=count,
                reader=metadata.reader,
                heuristic=decision.name,
                color_correction=metadata.properties.get("extract_patch.color_correction"),
                elapsed_seconds=elapsed,
                timings={
                    "segmentation": segment_time,
                    **timings,
                    "patches_per_second": count / max(elapsed, 1e-9),
                },
            )
    except Exception as exc:
        raise_if_system_error(exc)
        result = SlideResult(
            slide_id=spec.slide_id,
            status="failed",
            elapsed_seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
    return SlideWorkResult(result=result, patches=collector.records)


def extract(
    specs: Iterable[SlideSpec],
    config: AppConfig,
    *,
    run_id: str | None = None,
) -> RunResult:
    run_id = run_id or make_run_id("extract")
    output_root = Path(config.output.root)
    output_root.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(Path(config.logging.root), run_id, config)
    logger.logger.info(
        "Starting extraction; inputs will be resolved and inspected incrementally"
    )
    results_by_id: dict[str, SlideResult] = {}
    ordered_ids: list[str] = []
    resolved = 0

    def iter_work_items():
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
            slide_output = output_root / spec.slide_id
            patch_count = (
                None
                if config.output.overwrite
                else _complete_index_patch_count(slide_output, config.output.mode)
            )
            if patch_count is not None and _preview_is_complete(
                config, spec.slide_id, patch_count
            ):
                result = SlideResult(spec.slide_id, "skipped", patch_count=patch_count)
                logger.record_slide(result)
                results_by_id[spec.slide_id] = result
                continue
            completed = (
                {}
                if config.output.overwrite
                else _discover_completed_outputs(slide_output, config.output.mode)
            )
            if completed:
                logger.logger.info(
                    "slide=%s resuming_from_outputs=%d mode=%s",
                    spec.slide_id,
                    len(completed),
                    config.output.mode,
                )
            yield spec, completed

    def record_work_result(work_result: SlideWorkResult) -> None:
        for record in work_result.patches:
            logger.record_patch(work_result.result.slide_id, record)
        logger.record_slide(work_result.result)
        results_by_id[work_result.result.slide_id] = work_result.result

    try:
        with running_qc_service(config, logger.logger) as qc_client:
            process_map(
                _extract_slide,
                iter_work_items(),
                config,
                qc_client,
                max_workers=config.parallel.slide_workers,
                process_name="extract-wsi",
                on_result=record_work_result,
            )
            logger.logger.info("Input stream complete: resolved=%d", resolved)
            results = [results_by_id[slide_id] for slide_id in ordered_ids]
            _chmod_extract_outputs(config, output_root)
            summary = logger.finalize(results)
    except BaseException:
        try:
            _chmod_extract_outputs(config, output_root)
        except OSError:
            logger.logger.exception("Could not apply permissions to partial outputs")
        logger.abort("System-level extraction failure; all WSI workers were stopped")
        raise
    return RunResult(
        run_id=run_id,
        status="success" if summary["status"] == "success" else "partial",
        slides=results,
        output_root=output_root,
        log_dir=logger.path,
    )
