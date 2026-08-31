from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

from PIL import Image

from ..models import SlideMetadata
from .base import Location, ReaderBase, Size, normalized_metadata, preserve_aspect, validate_size


def _load_openslide() -> ModuleType:
    try:
        return importlib.import_module("openslide")
    except ImportError as error:
        raise ImportError(
            "OpenSlide backend requested, but the 'openslide' package is not installed"
        ) from error


class OpenSlideReader(ReaderBase):
    def __init__(self, path: str | Path, *, module: ModuleType | None = None) -> None:
        self.path = Path(path)
        openslide = module or _load_openslide()
        constructor = getattr(openslide, "OpenSlide", None)
        if constructor is None:
            constructor = getattr(openslide, "open_slide", None)
        if constructor is None:
            raise AttributeError("OpenSlide module does not expose OpenSlide or open_slide")
        # For POH/DICOM inputs this is the resolved volume entrypoint (normally 4_1.dcm).
        self._slide = constructor(str(self.path))
        self._closed = False
        try:
            self._metadata = normalized_metadata(self._slide, "openslide")
        except Exception:
            self.close()
            raise

    @property
    def metadata(self) -> SlideMetadata:
        return self._metadata

    def thumbnail(self, size: Size) -> Image.Image:
        size = validate_size(size)
        image = self._slide.get_thumbnail(size)
        return preserve_aspect(image, size)

    def read_region(self, location: Location, level: int, size: Size) -> Image.Image:
        return self._slide.read_region(
            (int(location[0]), int(location[1])), int(level), validate_size(size)
        )

    def close(self) -> None:
        if not self._closed:
            close = getattr(self._slide, "close", None)
            if close is not None:
                close()
            self._closed = True


OpenSlide = OpenSlideReader

__all__ = ["OpenSlide", "OpenSlideReader"]
