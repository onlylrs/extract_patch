from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import Any

from PIL import Image

from ..models import SlideMetadata
from .base import Location, ReaderBase, Size, normalized_metadata, preserve_aspect, validate_size


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


def _enable_color_correction(slide: Any) -> None:
    method = getattr(slide, "apply_color_correction", None)
    if method is None:
        return
    try:
        method(True, "Real")
    except NotImplementedError:
        # ASlide exposes the common API for all formats, including formats whose
        # backend does not provide color correction.
        return


class ASlideReader(ReaderBase):
    def __init__(self, path: str | Path, *, module: ModuleType | None = None) -> None:
        self.path = Path(path)
        self._slide = _open(module or _load_aslide(), self.path)
        self._closed = False
        try:
            _enable_color_correction(self._slide)
            self._metadata = normalized_metadata(self._slide, "aslide")
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
        return preserve_aspect(image, size)

    def read_region(self, location: Location, level: int, size: Size) -> Image.Image:
        # Both adapters deliberately pass level-0 coordinates through unchanged.
        return self._slide.read_region(
            (int(location[0]), int(location[1])), int(level), validate_size(size)
        )

    def close(self) -> None:
        if not self._closed:
            close = getattr(self._slide, "close", None)
            if close is not None:
                close()
            self._closed = True


ASlide = ASlideReader

__all__ = ["ASlide", "ASlideReader"]
