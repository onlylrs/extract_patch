from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from ..models import SlideSpec


def _stage_key(spec: SlideSpec) -> str:
    identity = str(spec.source.resolve()).encode("utf-8")
    return f"{spec.slide_id}-{hashlib.sha256(identity).hexdigest()[:16]}"


def _copy_path(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination, symlinks=True)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination, follow_symlinks=False)


def _copy_spec(spec: SlideSpec, bundle: Path) -> SlideSpec:
    if spec.source.is_dir():
        staged_source = bundle / spec.source.name
        _copy_path(spec.source, staged_source)
        try:
            relative_entrypoint = spec.entrypoint.relative_to(spec.source)
        except ValueError as error:
            raise ValueError("Directory slide entrypoint must be inside its source") from error
        staged_entrypoint = staged_source / relative_entrypoint
        staged_sidecars = tuple(staged_source / path.relative_to(spec.source) for path in spec.sidecars)
    else:
        staged_source = bundle / spec.source.name
        _copy_path(spec.source, staged_source)
        staged_entrypoint = staged_source
        copied_sidecars: list[Path] = []
        for sidecar in spec.sidecars:
            destination = bundle / sidecar.name
            _copy_path(sidecar, destination)
            copied_sidecars.append(destination)
        staged_sidecars = tuple(copied_sidecars)

    return SlideSpec(
        source=staged_source,
        entrypoint=staged_entrypoint,
        slide_id=spec.slide_id,
        sidecars=staged_sidecars,
    )


@contextmanager
def stage_slide(
    spec: SlideSpec,
    root: str | Path,
    *,
    cleanup: bool = True,
) -> Iterator[SlideSpec]:
    """Copy a slide bundle under an exclusive Linux advisory lock.

    MRXS main files and their same-stem sidecar directories are placed in one
    temporary bundle before that bundle is renamed into its final location.
    """
    staging_root = Path(root).expanduser()
    staging_root.mkdir(parents=True, exist_ok=True)
    lock_root = staging_root / ".locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    key = _stage_key(spec)
    target = staging_root / key

    lock_path = lock_root / f"{key}.lock"
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        temporary = Path(tempfile.mkdtemp(prefix=f".{key}-", dir=staging_root))
        try:
            staged = _copy_spec(spec, temporary)
            if target.exists():
                shutil.rmtree(target)
            os.replace(temporary, target)
            staged = SlideSpec(
                source=target / staged.source.relative_to(temporary),
                entrypoint=target / staged.entrypoint.relative_to(temporary),
                slide_id=staged.slide_id,
                sidecars=tuple(target / path.relative_to(temporary) for path in staged.sidecars),
            )
            try:
                yield staged
            finally:
                if cleanup:
                    shutil.rmtree(target, ignore_errors=True)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


__all__ = ["stage_slide"]
