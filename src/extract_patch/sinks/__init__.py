from __future__ import annotations

from pathlib import Path

from ..config import OutputConfig, parse_permission_mode
from .base import EncodedPatch, PatchSink, patch_stem, prepare_image
from .files import FileSink, ImageFileSink, JpegSink, NoneSink, NullSink, PngSink
from .tar import TarSink, generate_tar_preview


def create_sink(
    config: OutputConfig, output_dir: Path | str | None = None
) -> PatchSink:
    destination = Path(config.root) if output_dir is None else Path(output_dir)
    permissions = parse_permission_mode(config.permissions)
    if config.mode == "jpeg":
        return JpegSink(
            destination,
            jpeg_quality=config.jpeg_quality,
            overwrite=config.overwrite,
        )
    if config.mode == "png":
        return PngSink(destination, overwrite=config.overwrite)
    if config.mode == "tar":
        return TarSink(
            destination,
            shard_max_count=config.shard_max_count,
            jpeg_quality=config.jpeg_quality,
            preview_enabled=config.tar_preview,
            preview_count=config.tar_preview_n,
            preview_seed=config.tar_preview_seed,
            overwrite=config.overwrite,
            permissions=permissions,
        )
    if config.mode == "none":
        return NoneSink(destination)
    raise ValueError(f"Unknown output mode: {config.mode}")


sink_from_config = create_sink
make_sink = create_sink

__all__ = [
    "EncodedPatch",
    "FileSink",
    "ImageFileSink",
    "JpegSink",
    "NoneSink",
    "NullSink",
    "PatchSink",
    "PngSink",
    "TarSink",
    "create_sink",
    "generate_tar_preview",
    "make_sink",
    "patch_stem",
    "prepare_image",
    "sink_from_config",
]
