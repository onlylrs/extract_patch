from __future__ import annotations

import os
import tarfile
import tempfile
import threading
from io import BytesIO
from pathlib import Path

from PIL import Image

from ..models import PatchPlan
from .base import EncodedPatch, PatchSink, patch_stem, prepare_image


class TarSink(PatchSink):
    """Sequential, atomic WebDataset-style tar shard writer."""

    def __init__(
        self,
        output_dir: Path | str,
        *,
        shard_max_count: int = 1000,
        jpeg_quality: int = 90,
        overwrite: bool = False,
    ):
        if shard_max_count <= 0:
            raise ValueError("shard_max_count must be positive")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_max_count = shard_max_count
        self.jpeg_quality = jpeg_quality
        self.overwrite = overwrite
        existing = sorted(self.output_dir.glob("shard-*.tar"))
        self._shard_index = 0
        if existing and not overwrite:
            try:
                self._shard_index = max(int(path.stem.rsplit("-", 1)[1]) for path in existing) + 1
            except (IndexError, ValueError) as exc:
                raise ValueError(f"Invalid existing shard name in {self.output_dir}") from exc
        self._count = 0
        self._archive: tarfile.TarFile | None = None
        self._temporary: Path | None = None
        self._final: Path | None = None
        self._closed = False
        self._lock = threading.Lock()

    def encode(self, image: Image.Image, plan: PatchPlan) -> EncodedPatch:
        stream = BytesIO()
        image = prepare_image(image, plan)
        image.convert("RGB").save(stream, format="JPEG", quality=self.jpeg_quality)
        return EncodedPatch(patch_stem(plan) + ".jpg", stream.getvalue())

    def _open_shard(self) -> None:
        final = self.output_dir / f"shard-{self._shard_index:06d}.tar"
        if final.exists() and not self.overwrite:
            raise FileExistsError(final)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{self.output_dir.name}-{final.stem}-",
            suffix=".tmp",
            dir=self.output_dir.parent,
        )
        os.close(fd)
        self._temporary = Path(temporary_name)
        self._final = final
        self._archive = tarfile.open(self._temporary, mode="w")
        self._count = 0

    def _finish_shard(self) -> None:
        if self._archive is None:
            return
        temporary = self._temporary
        final = self._final
        try:
            self._archive.close()
            assert temporary is not None and final is not None
            fd = os.open(temporary, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            if self.overwrite:
                os.replace(temporary, final)
            else:
                os.link(temporary, final)
                temporary.unlink()
        except BaseException:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise
        finally:
            self._archive = None
            self._temporary = None
            self._final = None
            self._count = 0
        self._shard_index += 1

    def publish(self, encoded: EncodedPatch) -> str:
        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot write to a closed sink")
            if self._archive is None:
                self._open_shard()
            info = tarfile.TarInfo(encoded.name)
            info.size = len(encoded.data)
            info.mtime = 0
            info.mode = 0o644
            assert self._archive is not None
            self._archive.addfile(info, BytesIO(encoded.data))
            shard_name = f"shard-{self._shard_index:06d}.tar"
            self._count += 1
            if self._count >= self.shard_max_count:
                self._finish_shard()
            return f"{shard_name}/{encoded.name}"

    def close(self) -> None:
        if self._closed:
            return
        self._finish_shard()
        self._closed = True

    def abort(self) -> None:
        if self._closed:
            return
        if self._archive is not None:
            self._archive.close()
        if self._temporary is not None:
            self._temporary.unlink(missing_ok=True)
        self._archive = None
        self._temporary = None
        self._final = None
        self._count = 0
        self._closed = True
