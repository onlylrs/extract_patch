from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from extract_patch.config import AppConfig
from extract_patch.filters.base import FilterDecision
from extract_patch.models import PatchPlan, SlideMetadata, SlideSpec
from extract_patch import preview


def _plan(index: int) -> PatchPlan:
    return PatchPlan(
        index=index,
        x=index * 10,
        y=0,
        level=0,
        read_size=10,
        output_size=10,
        region_fraction=1.0,
    )


class _Reader:
    def __init__(self) -> None:
        self.read_indices = []

    def read_region(self, location, level, size):
        self.read_indices.append(location[0] // 10)
        return Image.new("RGB", size, color=(255, 255, 255))


def test_filter_preview_plans_draws_only_post_filter_accepts(monkeypatch) -> None:
    config = AppConfig(post_filter_pipe=["qc"])
    config.parallel.max_inflight_patches = 2
    plans = [_plan(index) for index in range(5)]
    calls = []

    class _Pipeline:
        def run_many(self, items):
            calls.append([plan.index for _image, plan in items])
            return [
                FilterDecision(keep=plan.index % 2 == 0, reason="test")
                for _image, plan in items
            ]

    monkeypatch.setattr(
        preview.PatchFilterPipeline,
        "from_config",
        lambda _config: _Pipeline(),
    )
    reader = _Reader()

    kept = preview._filter_preview_plans(reader, plans, config)

    assert [plan.index for plan in kept] == [0, 2, 4]
    assert calls == [[0, 1], [2, 3], [4]]
    assert reader.read_indices == [0, 1, 2, 3, 4]


def test_filter_preview_plans_avoids_reads_without_post_filters() -> None:
    config = AppConfig()
    plans = [_plan(0), _plan(1)]
    reader = SimpleNamespace()

    assert preview._filter_preview_plans(reader, plans, config) is plans


@pytest.mark.parametrize("worker", [preview._preview_slide, preview._center_preview_slide])
def test_large_single_level_slide_skips_before_thumbnail(monkeypatch, tmp_path: Path, worker) -> None:
    metadata = SlideMetadata(
        dimensions=(92928, 89856),
        level_dimensions=((92928, 89856), (23232, 22464), (5808, 5616)),
        level_downsamples=(1.0, 4.0, 16.0),
        mpp=0.5,
        reader="aslide",
    )

    @contextmanager
    def fake_open_reader(_spec, _config):
        yield SimpleNamespace(
            metadata=metadata,
            source_level_count=1,
            thumbnail=lambda _size: pytest.fail("thumbnail should not be read"),
        )

    monkeypatch.setattr(preview, "open_reader", fake_open_reader)
    spec = SlideSpec(tmp_path / "slide.tif", tmp_path / "slide.tif", "slide")

    result = worker(spec, AppConfig(), tmp_path)

    assert result.status == "skipped"
    assert "source has one level" in result.error
    assert not (tmp_path / "slide").exists()


def test_large_slide_with_real_pyramid_is_not_skipped() -> None:
    metadata = SlideMetadata(
        dimensions=(92928, 89856),
        level_dimensions=((92928, 89856), (23232, 22464)),
        level_downsamples=(1.0, 4.0),
        mpp=0.5,
        reader="aslide",
    )
    reader = SimpleNamespace(metadata=metadata, source_level_count=2)

    assert preview._large_single_level_skip_reason(reader) is None
