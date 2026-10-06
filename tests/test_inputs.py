from __future__ import annotations

import logging
import importlib.util
from pathlib import Path

import pytest

from extract_patch.inputs import iter_inputs, parse_inputs


def _write(root: Path, name: str, content: bytes = b"slide") -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_identical_same_name_slides_warn_and_continue(tmp_path, caplog):
    first = _write(tmp_path / "a", "slide.svs")
    duplicate = _write(tmp_path / "b", "slide.svs")
    middle = _write(tmp_path, "middle.svs", b"middle")
    last = _write(tmp_path, "last.svs", b"last")
    listing = tmp_path / "inputs.txt"
    listing.write_text("\n".join(str(p.relative_to(tmp_path)) for p in [first, middle, duplicate, last]))

    specs = parse_inputs(listing, root=tmp_path)

    assert [spec.entrypoint for spec in specs] == [first, middle, last]
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
    assert str(first) in caplog.text
    assert str(duplicate) in caplog.text
    assert "identical SHA-256" in caplog.text


def test_repeated_path_warns_and_skips(tmp_path, caplog):
    path = _write(tmp_path, "slide.svs")

    assert len(parse_inputs([path, path])) == 1
    assert "Duplicate WSI path" in caplog.text


def test_same_name_different_content_is_not_silently_discarded(tmp_path):
    first = _write(tmp_path / "a", "slide.svs", b"first")
    second = _write(tmp_path / "b", "slide.svs", b"other")

    with pytest.raises(ValueError, match="different content"):
        parse_inputs([first, second])


@pytest.mark.parametrize("format", ["mrxs", "dicom"])
@pytest.mark.parametrize("identical", [True, False])
def test_multifile_slides_compare_companion_content(tmp_path, format, identical):
    paths = []
    for index, parent in enumerate([tmp_path / "a", tmp_path / "b"]):
        content = b"volume" if identical or index == 0 else b"changed"
        if format == "mrxs":
            paths.append(_write(parent, "slide.mrxs"))
            _write(parent, "slide/nested/data.dat", content)
        else:
            _write(parent, "slide/4_1.dcm")
            _write(parent, "slide/4_2.dcm", content)
            paths.append(parent / "slide")

    if identical:
        assert len(parse_inputs(paths)) == 1
    else:
        with pytest.raises(ValueError, match="different content"):
            parse_inputs(paths)


def test_noncolliding_inputs_remain_lazy_and_do_not_hash(tmp_path, monkeypatch):
    from extract_patch import inputs

    first = _write(tmp_path, "first.svs")
    second = tmp_path / "second.svs"
    monkeypatch.setattr(inputs, "_slide_checksum", lambda _: pytest.fail("unexpected checksum"))
    specs = iter_inputs([first, second])

    assert next(specs).entrypoint == first
    second.write_bytes(b"second")
    assert next(specs).entrypoint == second
    with pytest.raises(StopIteration):
        next(specs)


@pytest.mark.parametrize("mode", ["extract", "preview", "center-preview"])
def test_cli_duplicate_warning_is_in_run_log_and_batch_succeeds(tmp_path, monkeypatch, mode):
    from extract_patch import pipeline, preview
    from extract_patch.models import SlideResult, SlideWorkResult

    script = Path(__file__).resolve().parents[1] / "extract_patches.py"
    module_spec = importlib.util.spec_from_file_location("extract_cli", script)
    cli = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(cli)
    first = _write(tmp_path / "a", "slide.svs")
    duplicate = _write(tmp_path / "b", "slide.svs")
    last = _write(tmp_path, "last.svs", b"last")
    submitted = []

    def fake_process_map(_worker, items, *_args, on_result, **_kwargs):
        results = []
        for item in items:
            spec = item[0] if mode == "extract" else item
            submitted.append(spec.entrypoint)
            result = SlideResult(spec.slide_id, "success")
            on_result(SlideWorkResult(result) if mode == "extract" else result)
            results.append(result)
        return results

    monkeypatch.setattr(pipeline if mode == "extract" else preview, "process_map", fake_process_map)
    monkeypatch.delenv("EXTRACT_PATCH_LOG_PATH", raising=False)
    monkeypatch.delenv("EXTRACT_PATCH_STDOUT_LOGGED", raising=False)
    run_id = f"duplicate_test_{mode}"
    args = [str(first), str(duplicate), str(last), "--run-id", run_id,
            "--output", str(tmp_path / "output"),
            "--set", f"logging.root={tmp_path / 'logs'}"]
    if mode != "extract":
        args.append(f"--{mode}")

    assert cli.main(args) == 0
    assert submitted == [first, last]
    log = (tmp_path / "logs" / f"{run_id}.log").read_text()
    assert "identical SHA-256" in log
    assert "summary status=success slides=2" in log
