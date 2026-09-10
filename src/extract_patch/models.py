from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class SlideSpec:
    source: Path
    entrypoint: Path
    slide_id: str
    sidecars: tuple[Path, ...] = ()


@dataclass(frozen=True)
class SlideMetadata:
    dimensions: tuple[int, int]
    level_dimensions: tuple[tuple[int, int], ...]
    level_downsamples: tuple[float, ...]
    mpp: float | None
    reader: str
    properties: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RegionMask:
    mask: np.ndarray
    thumbnail_size: tuple[int, int]
    slide_size: tuple[int, int]
    metadata: dict[str, Any] = field(default_factory=dict)


DecisionStatus = Literal["accepted", "not_applicable", "rejected", "error"]


@dataclass(frozen=True)
class HeuristicDecision:
    name: str
    status: DecisionStatus
    confidence: float
    reason: str
    region: RegionMask | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PatchPlan:
    index: int
    x: int
    y: int
    level: int
    read_size: int
    output_size: int
    region_fraction: float
    mpp: float | None = None


@dataclass(frozen=True)
class PatchRecord:
    index: int
    x: int
    y: int
    level: int
    size: int
    output: str
    status: Literal["success", "failed", "skipped", "filtered"]
    error: str | None = None


@dataclass
class SlideResult:
    slide_id: str
    status: Literal["success", "failed", "skipped"]
    patch_count: int = 0
    reader: str | None = None
    heuristic: str | None = None
    color_correction: str | None = None
    elapsed_seconds: float = 0.0
    error: str | None = None
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class SlideWorkResult:
    result: SlideResult
    patches: list[PatchRecord] = field(default_factory=list)


@dataclass
class RunResult:
    run_id: str
    status: Literal["success", "partial", "failed"]
    slides: list[SlideResult]
    output_root: Path
    log_dir: Path
