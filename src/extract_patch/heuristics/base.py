from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, TypeVar

import cv2
import numpy as np

from ..models import HeuristicDecision, RegionMask

Config = Mapping[str, Any]
H = TypeVar("H", bound="Heuristic")

_REGISTRY: dict[str, type["Heuristic"]] = {}


def register(cls: type[H]) -> type[H]:
    """Register a heuristic class by its stable configuration name."""
    if not cls.name:
        raise ValueError("A heuristic must define a non-empty name")
    if cls.name in _REGISTRY and _REGISTRY[cls.name] is not cls:
        raise ValueError(f"Heuristic already registered: {cls.name}")
    _REGISTRY[cls.name] = cls
    return cls


def registry() -> dict[str, type["Heuristic"]]:
    return dict(_REGISTRY)


def get_heuristic(name: str) -> "Heuristic":
    try:
        return _REGISTRY[name]()
    except KeyError as exc:
        raise KeyError(f"Unknown heuristic {name!r}; available: {sorted(_REGISTRY)}") from exc


def prepare_thumbnail(thumbnail: np.ndarray) -> np.ndarray:
    image = np.asarray(thumbnail)
    if image.ndim == 2:
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGBA2RGB)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("thumbnail must be HxW, HxWx3, or HxWx4")
    if image.shape[0] < 8 or image.shape[1] < 8:
        raise ValueError("thumbnail dimensions must both be at least 8 pixels")
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and float(np.nanmax(image)) <= 1.0:
            image = image * 255.0
        image = np.nan_to_num(image, nan=255.0, posinf=255.0, neginf=0.0)
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def foreground_seed(image: np.ndarray, *, color_delta: float = 10.0) -> np.ndarray:
    """Estimate non-background pixels against the bright thumbnail border."""
    image = prepare_thumbnail(image)
    height, width = image.shape[:2]
    band = max(2, int(round(min(height, width) * 0.035)))
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    border = np.concatenate(
        (
            lab[:band].reshape(-1, 3),
            lab[-band:].reshape(-1, 3),
            lab[:, :band].reshape(-1, 3),
            lab[:, -band:].reshape(-1, 3),
        )
    )
    bright = border[border[:, 0] >= np.percentile(border[:, 0], 55)]
    background = np.median(bright if len(bright) else border, axis=0)
    delta = np.sqrt(
        0.25 * (lab[:, :, 0] - background[0]) ** 2
        + (lab[:, :, 1] - background[1]) ** 2
        + (lab[:, :, 2] - background[2]) ** 2
    )
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    black_artifact = np.all(image < 35, axis=2)
    seed = ((delta >= color_delta) | (gray < min(210, np.percentile(gray, 20)))).astype(
        np.uint8
    )
    seed[black_artifact] = 0
    return seed


def odd_size(value: float, minimum: int = 3) -> int:
    size = max(minimum, int(round(value)))
    return size if size % 2 else size + 1


def circle_mask(shape: tuple[int, int], x: float, y: float, radius: float) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.circle(
        mask,
        (int(round(x)), int(round(y))),
        max(1, int(round(radius))),
        1,
        thickness=cv2.FILLED,
    )
    return mask.astype(bool)


def make_region(
    mask: np.ndarray,
    thumbnail: np.ndarray,
    slide_size: tuple[int, int] | None,
    metadata: Mapping[str, Any] | None = None,
) -> RegionMask:
    height, width = prepare_thumbnail(thumbnail).shape[:2]
    output = np.asarray(mask)
    if output.ndim != 2 or output.shape != (height, width):
        raise ValueError(
            f"mask shape {output.shape} does not match thumbnail shape {(height, width)}"
        )
    resolved_slide_size = slide_size or (width, height)
    if len(resolved_slide_size) != 2 or min(resolved_slide_size) <= 0:
        raise ValueError("slide_size must be a positive (width, height) pair")
    return RegionMask(
        mask=np.ascontiguousarray(output, dtype=bool),
        thumbnail_size=(width, height),
        slide_size=(int(resolved_slide_size[0]), int(resolved_slide_size[1])),
        metadata=dict(metadata or {}),
    )


def decision(
    name: str,
    status: str,
    reason: str,
    *,
    confidence: float = 0.0,
    region: RegionMask | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> HeuristicDecision:
    return HeuristicDecision(
        name=name,
        status=status,  # type: ignore[arg-type]
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        reason=reason,
        region=region,
        metrics=dict(metrics or {}),
    )


class Heuristic(ABC):
    name = ""

    @abstractmethod
    def is_applicable(
        self, thumbnail: np.ndarray, config: Config | None = None
    ) -> HeuristicDecision:
        """Return accepted when detection should run, otherwise not_applicable."""

    @abstractmethod
    def detect(
        self,
        thumbnail: np.ndarray,
        slide_size: tuple[int, int] | None = None,
        config: Config | None = None,
    ) -> HeuristicDecision:
        """Return the final structured detection decision."""

    def __call__(
        self,
        thumbnail: np.ndarray,
        slide_size: tuple[int, int] | None = None,
        config: Config | None = None,
    ) -> HeuristicDecision:
        try:
            applicable = self.is_applicable(thumbnail, config)
            if applicable.status != "accepted":
                return applicable
            return self.detect(thumbnail, slide_size, config)
        except Exception as exc:
            return decision(
                self.name,
                "error",
                f"{type(exc).__name__}: {exc}",
                metrics={"exception_type": type(exc).__name__},
            )
