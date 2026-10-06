from __future__ import annotations

from collections.abc import Iterator, Sequence
import hashlib
import logging
from pathlib import Path

from .models import SlideSpec


WSI_EXTENSIONS = {
    ".svs",
    ".sdpc",
    ".kfb",
    ".mrxs",
    ".ndpi",
    ".tif",
    ".tiff",
    ".tmap",
    ".mds",
    ".mdsx",
    ".tron",
    ".isyntax",
    ".czi",
    ".dcm",
}


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _dicom_entrypoint(directory: Path) -> Path:
    preferred = directory / "4_1.dcm"
    if preferred.is_file():
        return preferred
    candidates = sorted(directory.glob("*.dcm"))
    volume = [path for path in candidates if path.name.startswith("4_")]
    if volume:
        return volume[0]
    if candidates:
        return candidates[0]
    raise ValueError(f"No DICOM files found in directory: {directory}")


def resolve_slide(path: str | Path) -> SlideSpec:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_dir():
        entrypoint = _dicom_entrypoint(source)
        return SlideSpec(source=source, entrypoint=entrypoint, slide_id=source.name)
    if source.suffix.lower() not in WSI_EXTENSIONS:
        raise ValueError(f"Unsupported WSI extension: {source.suffix or '<none>'}")
    sidecars: tuple[Path, ...] = ()
    companion = source.with_suffix("")
    if source.suffix.lower() == ".mrxs" and companion.is_dir():
        sidecars = (companion,)
    return SlideSpec(source=source, entrypoint=source, slide_id=source.stem, sidecars=sidecars)


def _iter_input_values(
    value: str | Path | Sequence[str | Path],
) -> Iterator[str | Path]:
    if isinstance(value, (str, Path)):
        candidate = Path(value)
        if candidate.suffix.lower() == ".txt" and candidate.is_file():
            with candidate.open(encoding="utf-8") as handle:
                for line in handle:
                    item = _strip_quotes(line)
                    if item:
                        yield item
        else:
            yield value
    else:
        yield from value


def _slide_checksum(spec: SlideSpec) -> str:
    """Hash the full slide, including DICOM volumes and MRXS sidecars."""
    digest = hashlib.sha256()
    roots = (spec.source,) if spec.source.is_dir() else (spec.entrypoint, *spec.sidecars)
    for index, root in enumerate(roots):
        files = sorted(path for path in root.rglob("*") if path.is_file()) if root.is_dir() else [root]
        for path in files:
            relative = path.relative_to(root).as_posix() if root.is_dir() else ""
            label = f"{index}:{relative}".encode("utf-8")
            digest.update(len(label).to_bytes(8, "big"))
            digest.update(label)
            digest.update(path.stat().st_size.to_bytes(8, "big"))
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def iter_inputs(
    value: str | Path | Sequence[str | Path],
    *,
    root: Path | None = None,
    logger: logging.Logger | None = None,
) -> Iterator[SlideSpec]:
    """Resolve inputs incrementally so large lists can start work immediately."""
    seen_entries: set[Path] = set()
    seen_ids: dict[str, SlideSpec] = {}
    checksums: dict[str, str] = {}
    logger = logger if logger is not None else logging.getLogger(__name__)
    resolved = 0
    for item in _iter_input_values(value):
        path = Path(_strip_quotes(str(item)))
        if not path.is_absolute() and root is not None:
            path = root / path
        spec = resolve_slide(path)
        if spec.entrypoint in seen_entries:
            logger.warning("Duplicate WSI path %s; skipping", spec.entrypoint)
            continue
        prior = seen_ids.get(spec.slide_id)
        if prior is not None:
            if spec.slide_id not in checksums:
                checksums[spec.slide_id] = _slide_checksum(prior)
            if _slide_checksum(spec) == checksums[spec.slide_id]:
                logger.warning(
                    "Duplicate WSI slide_id=%s with identical SHA-256 content: %s; "
                    "keeping %s and skipping duplicate",
                    spec.slide_id, spec.entrypoint, prior.entrypoint,
                )
                seen_entries.add(spec.entrypoint)
                continue
            raise ValueError(
                f"Duplicate slide_id {spec.slide_id!r} with different content: "
                f"{prior.entrypoint} and {spec.entrypoint}"
            )
        seen_entries.add(spec.entrypoint)
        seen_ids[spec.slide_id] = spec
        resolved += 1
        yield spec
    if not resolved:
        raise ValueError("No WSI inputs resolved")


def parse_inputs(
    value: str | Path | Sequence[str | Path],
    *,
    root: Path | None = None,
) -> list[SlideSpec]:
    """Resolve all inputs, retaining the original list-returning API."""
    return list(iter_inputs(value, root=root))
