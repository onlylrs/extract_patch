from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .base import (
    Config,
    Heuristic,
    circle_mask,
    decision,
    foreground_seed,
    make_region,
    prepare_thumbnail,
    register,
)


def _edge_fixture_mask(image: np.ndarray, config: Config) -> tuple[np.ndarray, int]:
    """Find large near-black components connected to a vertical canvas edge."""
    height, width = image.shape[:2]
    threshold = int(config.get("fixture_black_threshold", 70))
    dark = np.all(image <= threshold, axis=2).astype(np.uint8)
    close_size = max(
        3, int(round(width * float(config.get("fixture_close_fraction", 0.004))))
    )
    if close_size % 2 == 0:
        close_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        dark, connectivity=8
    )
    minimum_area = width * width * float(config.get("min_fixture_area_fraction", 0.001))
    edge_margin = width * float(config.get("fixture_edge_fraction", 0.025))
    fixtures = np.zeros((height, width), dtype=np.uint8)
    fixture_count = 0
    for label in range(1, count):
        x, _y, component_width, _component_height, area = stats[label]
        touches_side = x <= edge_margin or x + component_width >= width - edge_margin
        if touches_side and area >= minimum_area:
            fixtures[labels == label] = 1
            fixture_count += 1

    margin = max(
        0, int(round(width * float(config.get("fixture_margin_fraction", 0.025))))
    )
    if fixtures.any() and margin > 0:
        guard = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1,) * 2)
        fixtures = cv2.dilate(fixtures, guard)
    return fixtures.astype(bool), fixture_count


def _fit_center_circle(
    image: np.ndarray,
    fixture_mask: np.ndarray,
    config: Config,
) -> dict[str, float]:
    height, width = image.shape[:2]
    prior_y_raw = float(config.get("center_y_fraction", 0.50)) * height
    vertical_extent = (
        float(config.get("analysis_vertical_extent_fraction", 0.85)) * width
    )
    crop_y = max(0, int(round(prior_y_raw - vertical_extent)))
    crop_bottom = min(height, int(round(prior_y_raw + vertical_extent)))
    analysis_image = image[crop_y:crop_bottom]
    analysis_fixtures = fixture_mask[crop_y:crop_bottom]

    analysis_width = min(width, int(config.get("max_analysis_width", 256)))
    scale = analysis_width / width

    evidence = foreground_seed(
        analysis_image,
        color_delta=float(config.get("color_delta", 5.0)),
    ).astype(np.float32)
    evidence[analysis_fixtures] = 0.0
    if scale < 1.0:
        evidence = cv2.resize(
            evidence,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
    sigma = max(
        1.0, evidence.shape[1] * float(config.get("density_sigma_ratio", 0.005))
    )
    evidence = cv2.GaussianBlur(evidence, (0, 0), sigmaX=sigma, sigmaY=sigma)

    prior_x = float(config.get("center_x_fraction", 0.50)) * width * scale
    prior_y = (prior_y_raw - crop_y) * scale
    prior_radius = float(config.get("radius_fraction", 0.56)) * width * scale
    center_extent = float(config.get("center_search_fraction", 0.12)) * width * scale
    center_step = max(
        1.0, float(config.get("center_step_fraction", 0.03)) * width * scale
    )
    radius_extent = float(config.get("radius_search_fraction", 0.04)) * width * scale
    radius_step = max(
        1.0, float(config.get("radius_step_fraction", 0.02)) * width * scale
    )

    x_offsets = np.arange(
        -center_extent, center_extent + center_step * 0.5, center_step
    )
    y_offsets = np.arange(
        -center_extent, center_extent + center_step * 0.5, center_step
    )
    radius_offsets = np.arange(
        -radius_extent, radius_extent + radius_step * 0.5, radius_step
    )
    yy, xx = np.ogrid[: evidence.shape[0], : evidence.shape[1]]
    core_ratio = float(config.get("core_radius_ratio", 1.0))
    outer_inner = float(config.get("outer_inner_radius_ratio", 1.04))
    outer_outer = float(config.get("outer_outer_radius_ratio", 1.25))
    center_prior_weight = float(config.get("center_prior_weight", 0.006))
    radius_prior_weight = float(config.get("radius_prior_weight", 0.004))

    best: tuple[float, float, float, float, float, float, float] | None = None
    for y_offset in y_offsets:
        y = prior_y + y_offset
        for x_offset in x_offsets:
            x = prior_x + x_offset
            distance = np.hypot(xx - x, yy - y)
            normalized_shift = (x_offset * x_offset + y_offset * y_offset) / max(
                1.0, center_extent * center_extent
            )
            for radius_offset in radius_offsets:
                radius = prior_radius + radius_offset
                core = distance <= radius * core_ratio
                annulus = (distance >= radius * outer_inner) & (
                    distance <= radius * outer_outer
                )
                if not core.any() or not annulus.any():
                    continue
                inner_density = float(evidence[core].mean())
                outer_density = float(evidence[annulus].mean())
                contrast = inner_density - outer_density
                normalized_radius_shift = (radius_offset / max(1.0, radius_extent)) ** 2
                score = (
                    contrast
                    - center_prior_weight * normalized_shift
                    - radius_prior_weight * normalized_radius_shift
                )
                candidate = (
                    score,
                    contrast,
                    x,
                    y,
                    radius,
                    inner_density,
                    outer_density,
                )
                if best is None or candidate[0] > best[0]:
                    best = candidate

    if best is None:
        best = (0.0, 0.0, prior_x, prior_y, prior_radius, 0.0, 0.0)
    (
        _score,
        contrast,
        fitted_x,
        fitted_y,
        fitted_radius,
        inner_density,
        outer_density,
    ) = best
    minimum_contrast = float(config.get("min_fit_contrast", 0.004))
    strong_contrast = max(
        minimum_contrast + 1e-6,
        float(config.get("strong_fit_contrast", 0.04)),
    )
    reliability = float(
        np.clip(
            (contrast - minimum_contrast) / (strong_contrast - minimum_contrast),
            0.0,
            1.0,
        )
    )

    return {
        "x": (prior_x + reliability * (fitted_x - prior_x)) / scale,
        "y": (prior_y + reliability * (fitted_y - prior_y)) / scale + crop_y,
        "radius": (prior_radius + reliability * (fitted_radius - prior_radius)) / scale,
        "fitted_x": fitted_x / scale,
        "fitted_y": fitted_y / scale + crop_y,
        "fitted_radius": fitted_radius / scale,
        "fit_contrast": contrast,
        "fit_reliability": reliability,
        "inner_density": inner_density,
        "outer_density": outer_density,
        "evidence_fraction": float(evidence.mean()),
    }


@register
class QmhCenterCircleHeuristic(Heuristic):
    """Select QMH's central circular smear while excluding black edge fixtures."""

    name = "qmh_center_circle"

    def is_applicable(self, thumbnail: np.ndarray, config: Config | None = None):
        try:
            image = prepare_thumbnail(thumbnail)
        except Exception as exc:
            return decision(self.name, "error", f"Invalid thumbnail: {exc}")
        cfg = config or {}
        canvas_aspect = image.shape[1] / image.shape[0]
        metrics = {"canvas_aspect": canvas_aspect}
        if (
            not float(cfg.get("min_canvas_aspect", 0.30))
            <= canvas_aspect
            <= float(cfg.get("max_canvas_aspect", 0.60))
        ):
            return decision(
                self.name,
                "not_applicable",
                "Canvas geometry does not match the portrait QMH carrier",
                metrics=metrics,
            )
        return decision(
            self.name,
            "accepted",
            "Portrait QMH carrier geometry found",
            confidence=0.8,
            metrics=metrics,
        )

    def detect(
        self,
        thumbnail: np.ndarray,
        slide_size: tuple[int, int] | None = None,
        config: Config | None = None,
    ):
        image = prepare_thumbnail(thumbnail)
        cfg: dict[str, Any] = dict(config or {})
        fixture_mask, fixture_count = _edge_fixture_mask(image, cfg)
        fitted = _fit_center_circle(image, fixture_mask, cfg)
        mask = circle_mask(
            image.shape[:2],
            fitted["x"],
            fitted["y"],
            fitted["radius"],
        )
        mask &= ~fixture_mask
        metrics = {
            **fitted,
            "fixture_count": fixture_count,
            "fixture_fraction": float(fixture_mask.mean()),
            "mask_fraction": float(mask.mean()),
        }
        confidence = min(0.97, 0.72 + 0.23 * fitted["fit_reliability"])
        return decision(
            self.name,
            "accepted",
            "Selected the central QMH circle and excluded black edge fixtures",
            confidence=confidence,
            region=make_region(mask, image, slide_size, {"method": self.name}),
            metrics=metrics,
        )


def qmh_center_circle(
    thumbnail: np.ndarray,
    slide_size: tuple[int, int] | None = None,
    config: Config | None = None,
):
    return QmhCenterCircleHeuristic()(thumbnail, slide_size, config)
