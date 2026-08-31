from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Iterator

from ..config import ReaderConfig
from ..models import SlideSpec
from .aslide import ASlide, ASlideReader
from .base import SlideReader
from .openslide import OpenSlide, OpenSlideReader
from .staging import stage_slide


class ReaderOpenError(RuntimeError):
    pass


def _backend_order(config: ReaderConfig) -> tuple[str, ...]:
    backend = config.backend.strip().lower()
    if backend in {"aslide", "openslide"}:
        return (backend,)
    if backend != "auto":
        raise ValueError("reader.backend must be 'auto', 'aslide', or 'openslide'")

    fallback = config.fallback.strip().lower()
    if fallback in {"", "none", "disabled"}:
        return ("aslide",)
    if fallback not in {"aslide", "openslide"}:
        raise ValueError("reader.fallback must be 'aslide', 'openslide', or 'none'")
    return tuple(dict.fromkeys(("aslide", fallback)))


def _construct(backend: str, spec: SlideSpec) -> SlideReader:
    if backend == "aslide":
        return ASlideReader(spec.entrypoint)
    return OpenSlideReader(spec.entrypoint)


@contextmanager
def open_reader(spec: SlideSpec, config: ReaderConfig) -> Iterator[SlideReader]:
    """Open a configured reader, staging the complete slide bundle if requested."""
    staging = (
        stage_slide(spec, config.staging_root, cleanup=config.staging_cleanup)
        if config.staging_enabled
        else nullcontext(spec)
    )
    with staging as active_spec:
        failures: list[tuple[str, Exception]] = []
        reader: SlideReader | None = None
        for backend in _backend_order(config):
            try:
                reader = _construct(backend, active_spec)
                break
            except Exception as error:
                failures.append((backend, error))
        if reader is None:
            detail = "; ".join(f"{name}: {error}" for name, error in failures)
            raise ReaderOpenError(f"Unable to open {spec.entrypoint} ({detail})") from failures[-1][1]
        with reader:
            yield reader


create_reader = open_reader
reader_factory = open_reader

__all__ = [
    "ASlide",
    "ASlideReader",
    "OpenSlide",
    "OpenSlideReader",
    "ReaderOpenError",
    "SlideReader",
    "create_reader",
    "open_reader",
    "reader_factory",
    "stage_slide",
]
