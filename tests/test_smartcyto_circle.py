from __future__ import annotations

import cv2
import numpy as np

from extract_patch.heuristics.smartcyto_circle import candidate_metrics, feature_edge


def test_feature_edge_handles_float64_percentiles(monkeypatch):
    original = np.percentile

    def float64_percentile(*args, **kwargs):
        return np.asarray(original(*args, **kwargs), dtype=np.float64)

    monkeypatch.setattr(np, "percentile", float64_percentile)
    image = np.random.default_rng(7).integers(0, 256, (96, 96, 3), dtype=np.uint8)
    edge = feature_edge(image)
    assert edge.dtype == np.float32
    assert edge.shape == image.shape[:2]
    assert np.isfinite(edge).all()
    assert edge.max() > 0


def test_strong_stained_circle_can_pass_configured_confidence():
    image = np.full((160, 160, 3), 235, dtype=np.uint8)
    cv2.circle(image, (80, 80), 50, (190, 70, 160), cv2.FILLED)
    edge = feature_edge(image)
    candidate = candidate_metrics(image, edge, (80, 80, 50))
    assert candidate["boundary_coverage"] >= 0.65
    assert candidate["content_spread"] == 1.0
    assert 0.70 <= candidate["score"] <= 1.0


def test_foam_pad_replaces_interior_circle_without_clipping_stain():
    from extract_patch.heuristics.smartcyto_circle import (
        make_roi,
        select_stained_texture_pad_circle,
    )

    gray = np.random.default_rng(11).integers(130, 256, (240, 240), dtype=np.uint8)
    image = np.repeat(gray[:, :, None], 3, axis=2)
    cv2.circle(image, (120, 120), 85, (190, 70, 160), cv2.FILLED)
    edge = feature_edge(image)
    interior = candidate_metrics(image, edge, (120, 120, 45))
    recovered = select_stained_texture_pad_circle(image, edge, interior)
    assert recovered is not None
    yy, xx = np.ogrid[:240, :240]
    stain = (xx - 120) ** 2 + (yy - 120) ** 2 <= 85 ** 2
    roi = make_roi(image.shape[:2], [recovered], 1.0) > 0
    assert np.count_nonzero(stain & roi) / stain.sum() >= 0.99
    assert roi.mean() < 0.60
    assert select_stained_texture_pad_circle(image, edge, recovered) is None


def test_stained_pad_recovery_requires_textured_carrier():
    from extract_patch.heuristics.smartcyto_circle import select_stained_texture_pad_circle

    image = np.full((240, 240, 3), 235, dtype=np.uint8)
    cv2.circle(image, (120, 120), 85, (190, 70, 160), cv2.FILLED)
    assert select_stained_texture_pad_circle(image, feature_edge(image)) is None
