from __future__ import annotations

import importlib
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

from PIL import Image, ImageCms

from ..models import SlideMetadata
from .base import Location, ReaderBase, Size, normalized_metadata, preserve_aspect, validate_size


def _select_thumbnail_level(
    level_dimensions: tuple[tuple[int, int], ...],
    requested: Size,
) -> int:
    fitting = [
        level
        for level, dimensions in enumerate(level_dimensions)
        if dimensions[0] <= requested[0] and dimensions[1] <= requested[1]
    ]
    return fitting[0] if fitting else len(level_dimensions) - 1


def _load_aslide() -> ModuleType:
    errors: list[ImportError] = []
    for module_name in ("Aslide", "aslide"):
        try:
            return importlib.import_module(module_name)
        except ImportError as error:
            errors.append(error)
    raise ImportError(
        "ASlide backend requested, but neither 'Aslide' nor 'aslide' can be imported"
    ) from errors[-1]


def _open(module: ModuleType, path: Path) -> Any:
    for name in ("ASlide", "Aslide", "Slide", "OpenSlide"):
        constructor = getattr(module, name, None)
        if constructor is not None:
            return constructor(str(path))
    for name in ("open_slide", "open"):
        function = getattr(module, name, None)
        if function is not None:
            return function(str(path))
    raise AttributeError("ASlide module does not expose a supported slide constructor")


def _openslide_color_transforms(slide: Any) -> tuple[Any | None, Any | None]:
    backend = getattr(slide, "backend", None)
    if backend is None:
        backend = getattr(slide, "_backend", None)
    source_profile = getattr(backend, "color_profile", None)
    if source_profile is None:
        return None, None
    target_profile = ImageCms.createProfile("sRGB")
    intent = ImageCms.getDefaultIntent(source_profile)
    return (
        ImageCms.buildTransform(source_profile, target_profile, "RGB", "RGB", intent, 0),
        ImageCms.buildTransform(source_profile, target_profile, "RGBA", "RGBA", intent, 0),
    )


def _enable_color_correction(slide: Any) -> tuple[Any | None, Any | None, str]:
    method = getattr(slide, "apply_color_correction", None)
    if method is None:
        return None, None, "unavailable:no-api"
    try:
        method(True, "Real")
    except NotImplementedError:
        rgb_transform, rgba_transform = _openslide_color_transforms(slide)
        if rgb_transform is None:
            return None, None, "unavailable:no-icc-profile"
        return rgb_transform, rgba_transform, "enabled:openslide-icc-to-srgb"
    registry_entry = getattr(slide, "registry_entry", None)
    format_id = getattr(registry_entry, "format_id", "native")
    backend = getattr(slide, "backend", None)
    info_method = getattr(backend, "get_color_correction_info", None)
    if info_method is not None:
        info = info_method()
        if isinstance(info, dict) and not info.get("enabled", False):
            return None, None, f"unavailable:aslide-real-not-enabled:{format_id}"
    return None, None, f"enabled:aslide-real:{format_id}"


class ASlideReader(ReaderBase):
    def __init__(self, path: str | Path, *, module: ModuleType | None = None) -> None:
        self.path = Path(path)
        self._slide = _open(module or _load_aslide(), self.path)
        self._closed = False
        try:
            (
                self._rgb_color_transform,
                self._rgba_color_transform,
                self.color_correction,
            ) = _enable_color_correction(self._slide)
            metadata = normalized_metadata(self._slide, "aslide")
            properties = dict(metadata.properties)
            properties["extract_patch.color_correction"] = self.color_correction
            self._metadata = replace(metadata, properties=properties)
        except Exception:
            self.close()
            raise

    @property
    def metadata(self) -> SlideMetadata:
        return self._metadata

    def thumbnail(self, size: Size) -> Image.Image:
        size = validate_size(size)
        method = getattr(self._slide, "get_thumbnail", None)
        if method is None:
            method = getattr(self._slide, "thumbnail", None)
        if method is None:
            raise AttributeError("ASlide object does not support thumbnails")
        image = method(size)
        if image is None:
            raise TypeError("ASlide thumbnail method did not return an image")
        expected_aspect = self.metadata.dimensions[0] / self.metadata.dimensions[1]
        actual_aspect = image.width / image.height
        aspect_error = abs(actual_aspect / expected_aspect - 1.0)
        if aspect_error > 0.02:
            level = _select_thumbnail_level(self.metadata.level_dimensions, size)
            dimensions = self.metadata.level_dimensions[level]
            image = self._slide.read_region((0, 0), level, dimensions)
            if image is None:
                raise TypeError("ASlide read_region did not return a thumbnail image")
        image = preserve_aspect(image, size)
        return self._apply_color_transform(image)

    def read_region(self, location: Location, level: int, size: Size) -> Image.Image:
        # Both adapters deliberately pass level-0 coordinates through unchanged.
        image = self._slide.read_region(
            (int(location[0]), int(location[1])), int(level), validate_size(size)
        )
        return self._apply_color_transform(image)

    def _apply_color_transform(self, image: Image.Image) -> Image.Image:
        if image.mode == "RGBA" and self._rgba_color_transform is not None:
            return ImageCms.applyTransform(image, self._rgba_color_transform)
        if self._rgb_color_transform is not None:
            rgb = image if image.mode == "RGB" else image.convert("RGB")
            return ImageCms.applyTransform(rgb, self._rgb_color_transform)
        return image

    def close(self) -> None:
        if not self._closed:
            close = getattr(self._slide, "close", None)
            if close is not None:
                close()
            self._closed = True


ASlide = ASlideReader

__all__ = ["ASlide", "ASlideReader", "_select_thumbnail_level"]
