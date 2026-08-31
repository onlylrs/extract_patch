from __future__ import annotations

import hashlib
import heapq
import os
import tarfile
import tempfile
import threading
from io import BytesIO
from pathlib import Path

from PIL import Image

from ..models import PatchPlan
from .base import EncodedPatch, PatchSink, patch_stem, prepare_image


def _preview_score(name: str, seed: int) -> int:
    digest = hashlib.blake2b(
        f"{seed}:{name}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big")


def _consider_preview(heap, count: int, name: str, payload, seed: int) -> None:
    if count <= 0:
        return
    score = _preview_score(name, seed)
    candidate = (-score, name, payload)
    if len(heap) < count:
        heapq.heappush(heap, candidate)
    elif score < -heap[0][0]:
        heapq.heapreplace(heap, candidate)


def _write_preview_files(
    output_dir: Path,
    selected: list[tuple[str, bytes]],
) -> list[Path]:
    preview_dir = output_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    selected_names = {name for name, _data in selected}
    for existing in preview_dir.glob("*.jpg"):
        if existing.name not in selected_names:
            existing.unlink()

    outputs = []
    for name, data in selected:
        destination = preview_dir / name
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{name}-",
            suffix=".tmp",
            dir=preview_dir,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                os.fchmod(handle.fileno(), 0o644)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        outputs.append(destination)
    return outputs


def generate_tar_preview(
    output_dir: Path | str,
    *,
    count: int = 20,
    seed: int = 0,
) -> list[Path]:
    """Create a deterministic random JPEG preview from existing TAR shards."""
    output_dir = Path(output_dir)
    heap = []
    for shard in sorted(output_dir.glob("shard-*.tar")):
        with tarfile.open(shard, mode="r") as archive:
            for member in archive:
                if member.isfile() and member.name.lower().endswith((".jpg", ".jpeg")):
                    _consider_preview(
                        heap,
                        count,
                        member.name,
                        (shard, member.name),
                        seed,
                    )

    selected = []
    for _negative_score, name, (shard, member_name) in sorted(heap, reverse=True):
        with tarfile.open(shard, mode="r") as archive:
            handle = archive.extractfile(member_name)
            if handle is not None:
                selected.append((name, handle.read()))
    return _write_preview_files(output_dir, selected) if selected else []


class TarSink(PatchSink):
    """Sequential, atomic WebDataset-style tar shard writer."""

    def __init__(
        self,
        output_dir: Path | str,
        *,
        shard_max_count: int = 1000,
        jpeg_quality: int = 90,
        preview_enabled: bool = False,
        preview_count: int = 20,
        preview_seed: int = 0,
        overwrite: bool = False,
    ):
        if shard_max_count <= 0:
            raise ValueError("shard_max_count must be positive")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_max_count = shard_max_count
        self.jpeg_quality = jpeg_quality
        self.preview_enabled = preview_enabled
        self.preview_count = preview_count
        self.preview_seed = preview_seed
        self._preview_heap = []
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
            if self.preview_enabled:
                _consider_preview(
                    self._preview_heap,
                    self.preview_count,
                    encoded.name,
                    encoded.data,
                    self.preview_seed,
                )
            shard_name = f"shard-{self._shard_index:06d}.tar"
            self._count += 1
            if self._count >= self.shard_max_count:
                self._finish_shard()
            return f"{shard_name}/{encoded.name}"

    def close(self) -> None:
        if self._closed:
            return
        self._finish_shard()
        if self.preview_enabled and self._preview_heap:
            selected = [
                (name, data)
                for _negative_score, name, data in sorted(
                    self._preview_heap,
                    reverse=True,
                )
            ]
            _write_preview_files(self.output_dir, selected)
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
