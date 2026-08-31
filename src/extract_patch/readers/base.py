from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from PIL import Image

from ..models import SlideMetadata

Size = tuple[int, int]
Location = tuple[int, int]


@runtime_checkable
class SlideReader(Protocol):
    @property
    def metadata(self) -> SlideMetadata: ...

    def thumbnail(self, size: Size) -> Image.Image: ...

    def read_region(self, location: Location, level: int, size: Size) -> Image.Image: ...

    def close(self) -> None: ...

    def __enter__(self) -> "SlideReader": ...

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None: ...


class ReaderBase:
    """Shared context-manager behavior for concrete readers."""

    _closed = False

    def __enter__(self) -> "ReaderBase":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def validate_size(size: Size) -> Size:
    result = (int(size[0]), int(size[1]))
    if result[0] <= 0 or result[1] <= 0:
        raise ValueError("Image size must contain two positive integers")
    return result


def preserve_aspect(image: Image.Image, size: Size) -> Image.Image:
    """Ensure a backend thumbnail fits inside *size* without distortion."""
    size = validate_size(size)
    if image.width <= size[0] and image.height <= size[1]:
        return image
    result = image.copy()
    result.thumbnail(size)
    return result


def _as_pair(value: Any) -> tuple[float, ...]:
    if isinstance(value, str):
        parts = [part for part in re.split(r"[,;\\\s]+", value.strip()) if part]
    elif isinstance(value, (tuple, list)):
        parts = list(value)
    else:
        parts = [value]
    values: list[float] = []
    for part in parts:
        try:
            number = float(part)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number) and number > 0:
            values.append(number)
    return tuple(values)


def normalize_mpp(properties: Mapping[str, Any], direct: Any = None) -> float | None:
    """Return mean microns-per-pixel from common reader metadata spellings."""
    direct_values = _as_pair(direct)
    if direct_values:
        return sum(direct_values) / len(direct_values)

    lowered = {str(key).lower(): value for key, value in properties.items()}
    x_keys = ("openslide.mpp-x", "mpp-x", "mpp_x", "microns_per_pixel_x")
    y_keys = ("openslide.mpp-y", "mpp-y", "mpp_y", "microns_per_pixel_y")
    xy = [
        values[0]
        for keys in (x_keys, y_keys)
        for key in keys
        if (values := _as_pair(lowered.get(key)))
    ]
    if xy:
        return sum(xy) / len(xy)

    for key in ("aperio.mpp", "mpp", "microns_per_pixel", "resolution"):
        values = _as_pair(lowered.get(key))
        if values:
            return sum(values) / len(values)

    # DICOM PixelSpacing is expressed in millimetres.
    for key in ("pixelspacing", "pixel_spacing", "dicom.pixelspacing"):
        values = _as_pair(lowered.get(key))
        if values:
            return 1000.0 * sum(values) / len(values)
    return None


def normalized_metadata(slide: Any, reader: str) -> SlideMetadata:
    raw_properties = getattr(slide, "properties", {}) or {}
    if callable(raw_properties):
        raw_properties = raw_properties()
    properties = (
        {str(key): str(value) for key, value in raw_properties.items()}
        if isinstance(raw_properties, Mapping)
        else {}
    )

    raw_dimensions = getattr(slide, "dimensions")
    if callable(raw_dimensions):
        raw_dimensions = raw_dimensions()
    dimensions = tuple(int(value) for value in raw_dimensions)
    raw_levels = getattr(slide, "level_dimensions", None)
    if callable(raw_levels):
        raw_levels = raw_levels()
    raw_levels = raw_levels or (dimensions,)
    level_dimensions = tuple(tuple(int(value) for value in level) for level in raw_levels)
    raw_downsamples = getattr(slide, "level_downsamples", None)
    if callable(raw_downsamples):
        raw_downsamples = raw_downsamples()
    if raw_downsamples is None:
        raw_downsamples = tuple(dimensions[0] / level[0] for level in level_dimensions)
    level_downsamples = tuple(float(value) for value in raw_downsamples)

    direct_mpp = getattr(slide, "mpp", None)
    if direct_mpp is None:
        direct_mpp = getattr(slide, "microns_per_pixel", None)
    return SlideMetadata(
        dimensions=(dimensions[0], dimensions[1]),
        level_dimensions=level_dimensions,
        level_downsamples=level_downsamples,
        mpp=normalize_mpp(properties, direct_mpp),
        reader=reader,
        properties=properties,
    )
