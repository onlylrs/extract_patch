from __future__ import annotations

import os
import tempfile
from io import BytesIO
from pathlib import Path

from PIL import Image

from ..models import PatchPlan
from .base import EncodedPatch, PatchSink, patch_stem, prepare_image


class ImageFileSink(PatchSink):
    def __init__(
        self,
        output_dir: Path | str,
        *,
        image_format: str = "jpeg",
        jpeg_quality: int = 90,
        overwrite: bool = False,
    ):
        image_format = image_format.lower()
        if image_format not in {"jpeg", "png"}:
            raise ValueError("image_format must be jpeg or png")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_format = image_format
        self.jpeg_quality = jpeg_quality
        self.overwrite = overwrite

    @property
    def extension(self) -> str:
        return ".jpg" if self.image_format == "jpeg" else ".png"

    def encode(self, image: Image.Image, plan: PatchPlan) -> EncodedPatch:
        stream = BytesIO()
        image = prepare_image(image, plan)
        if self.image_format == "jpeg":
            image.convert("RGB").save(stream, format="JPEG", quality=self.jpeg_quality)
        else:
            image.save(stream, format="PNG")
        return EncodedPatch(patch_stem(plan) + self.extension, stream.getvalue())

    def publish(self, encoded: EncodedPatch) -> str:
        destination = self.output_dir / encoded.name
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{self.output_dir.name}-patch-",
            suffix=".tmp",
            dir=self.output_dir.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                os.fchmod(handle.fileno(), 0o644)
                handle.write(encoded.data)
                handle.flush()
                os.fsync(handle.fileno())
            if self.overwrite:
                os.replace(temporary, destination)
            else:
                os.link(temporary, destination)
                temporary.unlink()
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return encoded.name

    def close(self) -> None:
        return None


class JpegSink(ImageFileSink):
    def __init__(
        self,
        output_dir: Path | str,
        *,
        jpeg_quality: int = 90,
        overwrite: bool = False,
    ):
        super().__init__(
            output_dir,
            image_format="jpeg",
            jpeg_quality=jpeg_quality,
            overwrite=overwrite,
        )


class PngSink(ImageFileSink):
    def __init__(self, output_dir: Path | str, *, overwrite: bool = False):
        super().__init__(output_dir, image_format="png", overwrite=overwrite)


class NoneSink(PatchSink):
    def __init__(self, output_dir: Path | str):
        self.output_dir = Path(output_dir)

    def encode(self, image: Image.Image, plan: PatchPlan) -> EncodedPatch:
        return EncodedPatch(patch_stem(plan), b"")

    def publish(self, encoded: EncodedPatch) -> str:
        return ""

    def close(self) -> None:
        return None


FileSink = ImageFileSink
NullSink = NoneSink
