from __future__ import annotations

from collections.abc import Sequence
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


def parse_inputs(
    value: str | Path | Sequence[str | Path],
    *,
    root: Path | None = None,
) -> list[SlideSpec]:
    raw_items: list[str | Path]
    if isinstance(value, (str, Path)):
        candidate = Path(value)
        if candidate.suffix.lower() == ".txt" and candidate.is_file():
            raw_items = [
                _strip_quotes(line)
                for line in candidate.read_text(encoding="utf-8").splitlines()
                if _strip_quotes(line)
            ]
        else:
            raw_items = [value]
    else:
        raw_items = list(value)

    specs: list[SlideSpec] = []
    seen_entries: set[Path] = set()
    seen_ids: dict[str, Path] = {}
    for item in raw_items:
        path = Path(_strip_quotes(str(item)))
        if not path.is_absolute() and root is not None:
            path = root / path
        spec = resolve_slide(path)
        if spec.entrypoint in seen_entries:
            continue
        prior = seen_ids.get(spec.slide_id)
        if prior is not None and prior != spec.entrypoint:
            raise ValueError(
                f"Duplicate slide_id {spec.slide_id!r}: {prior} and {spec.entrypoint}"
            )
        seen_entries.add(spec.entrypoint)
        seen_ids[spec.slide_id] = spec.entrypoint
        specs.append(spec)
    if not specs:
        raise ValueError("No WSI inputs resolved")
    return specs
