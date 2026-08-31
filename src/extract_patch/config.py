from __future__ import annotations

import copy
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ReaderConfig:
    backend: str = "auto"
    fallback: str = "openslide"
    thumbnail_width: int = 2048
    staging_enabled: bool = False
    staging_root: str = "/dev/shm/extract_patch"
    staging_cleanup: bool = True


@dataclass
class PatchingConfig:
    level: int = 0
    patch_size: int = 1200
    output_size: int = 1200
    stride: int = 1200
    min_region_fraction: float = 0.10
    inclusion: str = "fraction"
    max_patches_per_slide: int = 0
    sampling: str = "all"
    seed: int = 0
    mpp: float | None = None


@dataclass
class OutputConfig:
    root: str = "outputs/patches"
    mode: str = "jpeg"
    jpeg_quality: int = 90
    shard_max_count: int = 1000
    tar_preview: bool = False
    tar_preview_n: int = 20
    tar_preview_seed: int = 0
    overwrite: bool = False


@dataclass
class ParallelConfig:
    slide_workers: int = 1
    read_workers_per_slide: int = 4
    encode_workers: int = 2
    writer_workers: int = 2
    max_inflight_patches: int = 32
    max_inflight_bytes: int = 536_870_912
    opencv_threads: int = 1


@dataclass
class LoggingConfig:
    root: str = "logs"
    level: str = "INFO"


@dataclass
class PreviewConfig:
    root: str = "outputs/preview"
    n_patches: int = 8
    seed: int = 0


@dataclass
class AppConfig:
    reader: ReaderConfig = field(default_factory=ReaderConfig)
    patching: PatchingConfig = field(default_factory=PatchingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    preview: PreviewConfig = field(default_factory=PreviewConfig)
    heuristic_pipe: list[str] = field(default_factory=lambda: ["density"])
    heuristics: dict[str, dict[str, Any]] = field(default_factory=dict)
    post_filter_pipe: list[str] = field(default_factory=list)
    post_filters: dict[str, dict[str, Any]] = field(default_factory=dict)


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _coerce(raw: dict[str, Any]) -> AppConfig:
    allowed = {field.name for field in dataclasses.fields(AppConfig)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {', '.join(unknown)}")
    return AppConfig(
        reader=ReaderConfig(**raw.get("reader", {})),
        patching=PatchingConfig(**raw.get("patching", {})),
        output=OutputConfig(**raw.get("output", {})),
        parallel=ParallelConfig(**raw.get("parallel", {})),
        logging=LoggingConfig(**raw.get("logging", {})),
        preview=PreviewConfig(**raw.get("preview", {})),
        heuristic_pipe=list(raw.get("heuristic_pipe", ["density"])),
        heuristics=dict(raw.get("heuristics", {})),
        post_filter_pipe=list(raw.get("post_filter_pipe", [])),
        post_filters=dict(raw.get("post_filters", {})),
    )


def validate_config(config: AppConfig) -> None:
    if not config.heuristic_pipe:
        raise ValueError("heuristic_pipe cannot be empty")
    if any(not isinstance(name, str) or not name for name in config.post_filter_pipe):
        raise ValueError("post_filter_pipe entries must be non-empty names")
    if config.patching.patch_size <= 0 or config.patching.output_size <= 0:
        raise ValueError("patch sizes must be positive")
    if config.patching.stride <= 0:
        raise ValueError("stride must be positive")
    if config.patching.mpp is not None and config.patching.mpp <= 0:
        raise ValueError("patching.mpp must be positive or null")
    if not 0 <= config.patching.min_region_fraction <= 1:
        raise ValueError("min_region_fraction must be in [0, 1]")
    if config.output.mode not in {"jpeg", "png", "tar", "none"}:
        raise ValueError("output.mode must be jpeg, png, tar, or none")
    if config.output.tar_preview_n < 0:
        raise ValueError("output.tar_preview_n cannot be negative")
    for name in (
        "slide_workers",
        "read_workers_per_slide",
        "encode_workers",
        "writer_workers",
        "max_inflight_patches",
    ):
        if getattr(config.parallel, name) <= 0:
            raise ValueError(f"parallel.{name} must be positive")


def parse_override(value: str) -> tuple[list[str], Any]:
    if "=" not in value:
        raise ValueError(f"Override must be KEY=VALUE: {value}")
    key, raw = value.split("=", 1)
    return key.split("."), yaml.safe_load(raw)


def apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    result = copy.deepcopy(raw)
    for expression in overrides:
        keys, value = parse_override(expression)
        cursor = result
        for key in keys[:-1]:
            cursor = cursor.setdefault(key, {})
            if not isinstance(cursor, dict):
                raise ValueError(f"Cannot set nested override: {expression}")
        cursor[keys[-1]] = value
    return result


def load_config(path: Path | None = None, overrides: list[str] | None = None) -> AppConfig:
    defaults = dataclasses.asdict(AppConfig())
    supplied: dict[str, Any] = {}
    if path is not None:
        text = path.read_text(encoding="utf-8")
        supplied = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
        supplied = supplied or {}
    merged = apply_overrides(_merge(defaults, supplied), overrides or [])
    config = _coerce(merged)
    validate_config(config)
    return config


def config_dict(config: AppConfig) -> dict[str, Any]:
    return dataclasses.asdict(config)


def dump_config(config: AppConfig) -> str:
    return yaml.safe_dump(config_dict(config), sort_keys=False, allow_unicode=True)
