from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .config import PatchingConfig
from .models import PatchPlan, RegionMask, SlideMetadata


@dataclass(frozen=True)
class IntegralRegion:
    """Boolean region mask with constant-time rectangular coverage queries."""

    mask: np.ndarray
    integral: np.ndarray
    slide_size: tuple[int, int]

    @classmethod
    def from_region(cls, region: RegionMask) -> "IntegralRegion":
        mask = np.asarray(region.mask, dtype=bool)
        expected_shape = (region.thumbnail_size[1], region.thumbnail_size[0])
        if mask.ndim != 2 or mask.shape != expected_shape:
            raise ValueError(
                f"Region mask shape {mask.shape} does not match thumbnail size "
                f"{region.thumbnail_size}"
            )
        integral = np.pad(mask.astype(np.uint64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        return cls(mask=mask, integral=integral, slide_size=region.slide_size)

    def _mask_box(
        self, x: int, y: int, width: float, height: float
    ) -> tuple[int, int, int, int]:
        slide_width, slide_height = self.slide_size
        mask_height, mask_width = self.mask.shape
        x0 = max(0, min(mask_width, int(np.floor(x * mask_width / slide_width))))
        y0 = max(0, min(mask_height, int(np.floor(y * mask_height / slide_height))))
        x1 = max(0, min(mask_width, int(np.ceil((x + width) * mask_width / slide_width))))
        y1 = max(0, min(mask_height, int(np.ceil((y + height) * mask_height / slide_height))))
        return x0, y0, x1, y1

    def fraction(self, x: int, y: int, width: float, height: float) -> float:
        x0, y0, x1, y1 = self._mask_box(x, y, width, height)
        area = (x1 - x0) * (y1 - y0)
        if area <= 0:
            return 0.0
        total = (
            int(self.integral[y1, x1])
            - int(self.integral[y0, x1])
            - int(self.integral[y1, x0])
            + int(self.integral[y0, x0])
        )
        return float(total) / area

    def contains(self, x: float, y: float) -> bool:
        slide_width, slide_height = self.slide_size
        mask_height, mask_width = self.mask.shape
        column = min(mask_width - 1, max(0, int(x * mask_width / slide_width)))
        row = min(mask_height - 1, max(0, int(y * mask_height / slide_height)))
        return bool(self.mask[row, column])


def grid_coordinates(
    metadata: SlideMetadata, level: int, patch_size: int, stride: int
) -> Iterable[tuple[int, int]]:
    """Yield level-0 coordinates from a grid defined at ``level``."""
    if not 0 <= level < len(metadata.level_dimensions):
        raise ValueError(f"Invalid level {level}")
    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be positive")

    width, height = metadata.level_dimensions[level]
    downsample = metadata.level_downsamples[level]
    if downsample <= 0:
        raise ValueError(f"Invalid downsample {downsample} for level {level}")
    if patch_size > width or patch_size > height:
        return

    for target_y in range(0, height - patch_size + 1, stride):
        for target_x in range(0, width - patch_size + 1, stride):
            yield round(target_x * downsample), round(target_y * downsample)


def _select(
    plans: list[PatchPlan], maximum: int, sampling: str, seed: int
) -> list[PatchPlan]:
    if maximum <= 0 or len(plans) <= maximum:
        selected = plans
    elif sampling == "all":
        selected = plans[:maximum]
    elif sampling == "random":
        indices = np.random.default_rng(seed).choice(len(plans), size=maximum, replace=False)
        selected = [plans[index] for index in sorted(indices.tolist())]
    elif sampling == "uniform":
        indices = np.linspace(0, len(plans) - 1, num=maximum, dtype=int)
        selected = [plans[index] for index in indices]
    else:
        raise ValueError(f"Unknown sampling mode: {sampling}")

    return [
        PatchPlan(
            index=index,
            x=plan.x,
            y=plan.y,
            level=plan.level,
            read_size=plan.read_size,
            output_size=plan.output_size,
            region_fraction=plan.region_fraction,
        )
        for index, plan in enumerate(selected)
    ]


def plan_patches(
    metadata: SlideMetadata,
    region: RegionMask,
    config: PatchingConfig,
) -> list[PatchPlan]:
    if not 0 <= config.level < len(metadata.level_dimensions):
        raise ValueError(f"Invalid level {config.level}")
    if config.level >= len(metadata.level_downsamples):
        raise ValueError(f"Missing downsample for level {config.level}")
    if config.patch_size <= 0 or config.output_size <= 0 or config.stride <= 0:
        raise ValueError("patch size, output size, and stride must be positive")
    if not 0 <= config.min_region_fraction <= 1:
        raise ValueError("min_region_fraction must be in [0, 1]")
    if config.max_patches_per_slide < 0:
        raise ValueError("max_patches_per_slide cannot be negative")
    if config.inclusion not in {"center", "fraction", "full"}:
        raise ValueError(f"Unknown inclusion mode: {config.inclusion}")
    if config.sampling not in {"all", "random", "uniform"}:
        raise ValueError(f"Unknown sampling mode: {config.sampling}")
    if region.slide_size != metadata.dimensions:
        raise ValueError("Region mask and slide metadata dimensions differ")

    integral = IntegralRegion.from_region(region)
    downsample = metadata.level_downsamples[config.level]
    extent = config.patch_size * downsample
    candidates: list[PatchPlan] = []
    for x, y in grid_coordinates(
        metadata, config.level, config.patch_size, config.stride
    ):
        fraction = integral.fraction(x, y, extent, extent)
        if config.inclusion == "center":
            included = integral.contains(x + extent / 2, y + extent / 2)
        elif config.inclusion == "full":
            included = fraction == 1.0
        else:
            included = fraction >= config.min_region_fraction
        if included:
            candidates.append(
                PatchPlan(
                    index=len(candidates),
                    x=x,
                    y=y,
                    level=config.level,
                    read_size=config.patch_size,
                    output_size=config.output_size,
                    region_fraction=fraction,
                )
            )

    return _select(
        candidates,
        config.max_patches_per_slide,
        config.sampling,
        config.seed,
    )


class PatchPlanner:
    def __init__(self, config: PatchingConfig):
        self.config = config

    def plan(self, metadata: SlideMetadata, region: RegionMask) -> list[PatchPlan]:
        return plan_patches(metadata, region, self.config)
