from __future__ import annotations

import cv2
import numpy as np

from .base import (
    Config,
    Heuristic,
    circle_mask,
    decision,
    make_region,
    prepare_thumbnail,
    register,
)
from .density import DensityHeuristic
from .heatmaps import circle_detection_heatmap


def _radial_inner_circle(
    image: np.ndarray,
    artifact: np.ndarray,
    prior_x: float,
    prior_y: float,
    config: Config,
) -> tuple[tuple[float, float, float], dict[str, float]]:
    """Fit the faint inner pad from its radial stain cutoff.

    POH thumbnails contain a strong, tiled outer scanner edge and a much
    fainter inner pad edge. A constrained polar search integrates that weak
    cutoff over the full circumference instead of letting a short outer arc
    dominate a conventional Hough transform.
    """

    height, width = image.shape[:2]
    min_dim = min(height, width)
    rgb = image.astype(np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    chroma = np.hypot(lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0)
    blue_excess = rgb[:, :, 2] - (rgb[:, :, 0] + rgb[:, :, 1]) / 2.0

    signal = (
        np.maximum(blue_excess - float(config.get("radial_blue_floor", 0.5)), 0.0)
        + 0.30 * np.maximum(chroma - float(config.get("radial_chroma_floor", 1.5)), 0.0)
        + 0.08 * np.maximum(254.0 - gray, 0.0)
    )
    signal[artifact] = 0.0
    signal = cv2.GaussianBlur(
        signal.astype(np.float32),
        (0, 0),
        max(1.5, min_dim * float(config.get("radial_signal_sigma_ratio", 0.003))),
    )

    min_radius = int(round(min_dim * float(config.get("radial_min_radius", 0.27))))
    max_radius = int(round(min_dim * float(config.get("radial_max_radius", 0.36))))
    band = max(3, int(round(min_dim * float(config.get("radial_band_ratio", 0.012)))))
    angle_count = int(config.get("radial_angle_count", 72))
    center_extent = min_dim * float(config.get("radial_center_extent", 0.09))
    coarse_step = max(4, int(round(min_dim * float(config.get("radial_center_step", 0.012)))))
    prior_radius = min_dim * float(config.get("preferred_radius_fraction", 0.32))

    def evaluate(center_x: int, center_y: int):
        polar = cv2.warpPolar(
            signal,
            (max_radius + band + 2, angle_count),
            (float(center_x), float(center_y)),
            float(max_radius + band + 1),
            cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS + cv2.INTER_LINEAR,
        )
        radius_scale = polar.shape[1] / float(max_radius + band + 1)
        radii = np.arange(min_radius, max_radius + 1, 2, dtype=np.int32)
        columns = np.rint(radii * radius_scale).astype(np.int32)
        band_columns = max(2, int(round(band * radius_scale)))
        cumulative = np.pad(
            np.cumsum(polar, axis=1, dtype=np.float64),
            ((0, 0), (1, 0)),
        )
        inner = (
            cumulative[:, columns] - cumulative[:, columns - band_columns]
        ) / band_columns
        outer = (
            cumulative[:, columns + 1 + band_columns] - cumulative[:, columns + 1]
        ) / band_columns
        difference = inner - outer
        positive_support = np.mean(difference > 0.02, axis=0)
        mean_difference = np.mean(difference, axis=0)
        lower_support = np.percentile(difference, 35, axis=0)
        dispersion = np.std(difference, axis=0)
        prior_offset = np.hypot(center_x - prior_x, center_y - prior_y) / min_dim
        radius_offset = np.abs(radii - prior_radius) / min_dim
        scores = (
            lower_support
            + 0.50 * mean_difference
            - 0.30 * dispersion
            + 0.08 * positive_support
            - 0.10 * prior_offset
            - 0.04 * radius_offset
        )
        index = int(np.argmax(scores))
        return (
            float(scores[index]),
            float(positive_support[index]),
            float(mean_difference[index]),
            float(lower_support[index]),
            float(dispersion[index]),
            center_x,
            center_y,
            int(radii[index]),
        )

    coarse = []
    for center_y in range(
        int(round(prior_y - center_extent)),
        int(round(prior_y + center_extent)) + 1,
        coarse_step,
    ):
        for center_x in range(
            int(round(prior_x - center_extent)),
            int(round(prior_x + center_extent)) + 1,
            coarse_step,
        ):
            candidate = evaluate(center_x, center_y)
            if candidate is not None:
                coarse.append(candidate)

    if not coarse:
        fallback = (prior_x, prior_y, prior_radius)
        return fallback, {"inner_radial_geometry_fallback": 1.0}

    coarse_best = max(coarse)
    best = coarse_best
    _, _, _, _, _, coarse_x, coarse_y, _ = coarse_best
    fine_step = max(2, coarse_step // 4)
    for center_y in range(coarse_y - coarse_step, coarse_y + coarse_step + 1, fine_step):
        for center_x in range(coarse_x - coarse_step, coarse_x + coarse_step + 1, fine_step):
            candidate = evaluate(center_x, center_y)
            if candidate is not None and candidate > best:
                best = candidate

    (
        score,
        positive_support,
        mean_difference,
        lower_support,
        dispersion,
        center_x,
        center_y,
        radius,
    ) = best
    return (
        (float(center_x), float(center_y), float(radius)),
        {
            "inner_radial": 1.0,
            "inner_radial_score": float(score),
            "inner_radial_positive_support": positive_support,
            "inner_radial_mean_difference": mean_difference,
            "inner_radial_lower_support": lower_support,
            "inner_radial_dispersion": dispersion,
        },
    )


@register
class SerratedOuterCircleHeuristic(Heuristic):
    """Select content inside a serrated scanner ring, never the ring itself."""

    name = "serrated_outer_circle"

    def _analyze(self, thumbnail: np.ndarray, config: Config | None):
        image = prepare_thumbnail(thumbnail)
        cfg = config or {}
        height, width = image.shape[:2]
        min_dim = min(height, width)
        prior_x = width * float(cfg.get("prior_center_x_fraction", 0.50))
        prior_y = height * float(cfg.get("prior_center_y_fraction", 0.41))
        max_prior_distance = float(cfg.get("max_prior_center_distance", 0.14))
        rgb = image.astype(np.float32)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        dark = 255.0 - gray
        blue_excess = rgb[:, :, 2] - (rgb[:, :, 0] + rgb[:, :, 1]) / 2.0
        yy, xx = np.indices((height, width), dtype=np.float32)
        canvas_radius = np.hypot(xx - (width - 1) / 2, yy - (height - 1) / 2) / min_dim

        # POH's large scanner boundary is a sparse green/gray, tile-shaped ring near
        # the canvas edge. It must be present before this specialized rule can apply.
        outer_annulus = (
            (canvas_radius >= float(cfg.get("outer_annulus_min", 0.36)))
            & (canvas_radius <= float(cfg.get("outer_annulus_max", 0.53)))
        )
        artifact = outer_annulus & (dark >= float(cfg.get("outer_dark_threshold", 10.0)))
        outer_score = float(artifact.sum() / max(1, outer_annulus.sum()))
        metrics: dict[str, float] = {
            "outer_artifact_fraction": outer_score,
            "outer_radius": float(min_dim * 0.50),
        }
        if (
            outer_score < float(cfg.get("min_outer_artifact_fraction", 0.025))
            or outer_score > float(cfg.get("max_outer_artifact_fraction", 0.45))
        ):
            return image, None, metrics

        # Blue/cyan stain separates the cell-rich inner pad from the green scanner
        # ring. Very dark cells are admitted only away from the outer annulus.
        stain = (
            (blue_excess >= float(cfg.get("min_blue_excess", 1.5)))
            & (dark >= float(cfg.get("min_stain_darkness", 1.0)))
        ) | (
            (dark >= float(cfg.get("dark_cell_threshold", 22.0)))
            & (canvas_radius < float(cfg.get("dark_cell_max_radius", 0.43)))
        )
        stain = stain.astype(np.float32)
        stain[artifact] = 0

        # The weak inner circumference is a global radial cutoff. Fit it before
        # Hough, whose strongest POH candidates are usually arcs of the serrated
        # scanner edge.
        radial_circle, radial_metrics = _radial_inner_circle(
            image, artifact, prior_x, prior_y, cfg
        )
        metrics.update(radial_metrics)
        if (
            float(radial_metrics.get("inner_radial_score", 0.0))
            < float(cfg.get("min_inner_radial_score", 0.0))
            or float(radial_metrics.get("inner_radial_positive_support", 0.0))
            < float(cfg.get("min_inner_radial_positive_support", 0.65))
            or float(radial_metrics.get("inner_radial_mean_difference", 0.0))
            < float(cfg.get("min_inner_radial_mean_difference", 0.02))
        ):
            metrics["inner_radial_safety_rejected"] = 1.0
            return image, None, metrics
        radial_x, radial_y, radial_radius = radial_circle
        metrics.update(
            {
                "inner_x": radial_x,
                "inner_y": radial_y,
                "inner_radius": radial_radius,
                "inner_radius_fraction": radial_radius / min_dim,
                "inner_center_distance": float(
                    np.hypot(radial_x - width / 2, radial_y - height / 2) / min_dim
                ),
                "inner_prior_distance": float(
                    np.hypot(radial_x - prior_x, radial_y - prior_y) / min_dim
                ),
                "inner_fill": 0.0,
            }
        )
        return image, radial_circle, metrics

        # Prefer an actual circular boundary fit. Density centroids drift when one
        # side of the pad contains more cells, which is exactly the failure mode
        # this heuristic must avoid.
        hough_score = circle_detection_heatmap(image)
        max_detection = int(cfg.get("hough_max_dimension", 1024))
        hough_scale = min(1.0, max_detection / max(height, width))
        hough_image = (
            cv2.resize(
                hough_score.astype(np.uint8),
                None,
                fx=hough_scale,
                fy=hough_scale,
                interpolation=cv2.INTER_AREA,
            )
            if hough_scale < 1.0
            else hough_score.astype(np.uint8)
        )
        hough_image = cv2.GaussianBlur(hough_image, (0, 0), 2.0)
        circles = cv2.HoughCircles(
            hough_image,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(20, int(min(hough_image.shape[:2]) * 0.12)),
            param1=float(cfg.get("hough_param1", 40)),
            param2=float(cfg.get("hough_param2", 25)),
            minRadius=int(
                min(hough_image.shape[:2]) * float(cfg.get("hough_search_min_radius", 0.22))
            ),
            maxRadius=int(
                min(hough_image.shape[:2]) * float(cfg.get("hough_search_max_radius", 0.42))
            ),
        )
        metrics["hough_candidate_count"] = 0.0 if circles is None else float(len(circles[0]))
        if circles is not None:
            ranked_circles = []
            for candidate_index, (small_x, small_y, small_radius) in enumerate(circles[0]):
                x = float(small_x / hough_scale)
                y = float(small_y / hough_scale)
                radius = float(small_radius / hough_scale)
                radius_fraction = radius / min_dim
                center_distance = float(
                    np.hypot(x - width / 2, y - height / 2) / min_dim
                )
                prior_distance = float(np.hypot(x - prior_x, y - prior_y) / min_dim)
                ring_width = max(2.0, min_dim * float(cfg.get("hough_ring_width", 0.018)))
                radial = np.hypot(xx - x, yy - y)
                ring = np.abs(radial - radius) <= ring_width
                ring_score = float(np.mean(hough_score[ring])) if ring.any() else 0.0
                ring_stain = ring & (stain > 0)
                sectors = int(cfg.get("hough_angular_sectors", 36))
                if ring_stain.any():
                    angles = np.arctan2(yy[ring_stain] - y, xx[ring_stain] - x)
                    bins = np.floor((angles + np.pi) * sectors / (2 * np.pi)).astype(int)
                    angular_coverage = float(
                        len(np.unique(np.clip(bins, 0, sectors - 1))) / sectors
                    )
                else:
                    angular_coverage = 0.0
                if (
                    float(cfg.get("hough_min_radius", 0.22))
                    <= radius_fraction
                    <= float(cfg.get("hough_max_radius", 0.36))
                    and center_distance <= float(cfg.get("max_inner_center_distance", 0.32))
                    and prior_distance <= max_prior_distance
                    and ring_score >= float(cfg.get("hough_min_ring_score", 0.35))
                ):
                    preferred_radius = float(cfg.get("preferred_radius_fraction", 0.32))
                    rank = (
                        -candidate_index * 100.0
                        + angular_coverage * 10.0
                        + min(ring_score, 20.0) * 0.1
                        - abs(radius_fraction - preferred_radius) * 4.0
                        - prior_distance * 2.0
                    )
                    ranked_circles.append(
                        (
                            rank,
                            x,
                            y,
                            radius,
                            radius_fraction,
                            center_distance,
                            prior_distance,
                            ring_score,
                            angular_coverage,
                        )
                    )
            if ranked_circles:
                (
                    _rank,
                    x,
                    y,
                    radius,
                    radius_fraction,
                    center_distance,
                    prior_distance,
                    ring_score,
                    angular_coverage,
                ) = max(ranked_circles)
                metrics.update(
                    {
                        "inner_hough": 1.0,
                        "inner_x": x,
                        "inner_y": y,
                        "inner_radius": radius,
                        "inner_radius_fraction": radius_fraction,
                        "inner_center_distance": center_distance,
                        "inner_prior_distance": prior_distance,
                        "inner_hough_ring_score": ring_score,
                        "inner_hough_angular_coverage": angular_coverage,
                        "inner_fill": 0.0,
                    }
                )
                return image, (x, y, radius), metrics

        sigma = max(1.5, min_dim * float(cfg.get("inner_density_sigma_ratio", 0.018)))
        density = cv2.GaussianBlur(stain, (0, 0), sigma)
        maximum = float(density.max())
        metrics["inner_max_density"] = maximum
        if maximum <= 0:
            return image, None, metrics

        threshold = max(
            float(cfg.get("min_density_cutoff", 0.004)),
            maximum * float(cfg.get("inner_density_threshold_ratio", 0.055)),
        )
        mask = (density >= threshold).astype(np.uint8)
        close_size = max(3, int(round(min_dim * float(cfg.get("inner_close_ratio", 0.025)))))
        close_size += 1 - close_size % 2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image, None, metrics

        candidates = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            (enclosing_x, enclosing_y), enclosing_radius = cv2.minEnclosingCircle(contour)
            moments = cv2.moments(contour)
            if moments["m00"] <= 0:
                continue
            x = float(moments["m10"] / moments["m00"])
            y = float(moments["m01"] / moments["m00"])
            radius = float(np.sqrt(area / np.pi))
            bx, by, bw, bh = cv2.boundingRect(contour)
            perimeter = float(cv2.arcLength(contour, True))
            circularity = 4 * np.pi * area / max(1.0, perimeter * perimeter)
            fill = area / max(1.0, np.pi * enclosing_radius * enclosing_radius)
            area_fraction = area / (height * width)
            radius_fraction = radius / min_dim
            aspect = max(bw, bh) / max(1, min(bw, bh))
            center_distance = np.hypot(x - width / 2, y - height / 2) / min_dim
            prior_distance = np.hypot(x - prior_x, y - prior_y) / min_dim
            valid = (
                float(cfg.get("min_inner_area_fraction", 0.06))
                <= area_fraction
                <= float(cfg.get("max_inner_area_fraction", 0.58))
                and float(cfg.get("min_inner_radius_fraction", 0.16))
                <= radius_fraction
                <= float(cfg.get("max_inner_radius_fraction", 0.42))
                and aspect <= float(cfg.get("max_inner_aspect", 1.45))
                and fill >= float(cfg.get("min_inner_fill", 0.40))
                and circularity >= float(cfg.get("min_inner_circularity", 0.35))
                and center_distance <= float(cfg.get("max_inner_center_distance", 0.32))
                and prior_distance <= max_prior_distance
                and 0.50 / radius_fraction
                >= float(cfg.get("min_outer_to_inner_ratio", 1.12))
            )
            if valid:
                candidates.append(
                    (
                        area,
                        float(x),
                        float(y),
                        float(radius),
                        area_fraction,
                        radius_fraction,
                        aspect,
                        fill,
                        circularity,
                        float(center_distance),
                        float(prior_distance),
                        float(enclosing_x),
                        float(enclosing_y),
                        float(enclosing_radius),
                    )
                )
        metrics["inner_candidate_count"] = float(len(candidates))
        if not candidates:
            points = np.argwhere(
                (stain > 0)
                & (canvas_radius < float(cfg.get("sparse_canvas_radius", 0.44)))
            )
            min_points = max(64, int(height * width * float(cfg.get("min_stain_fraction", 0.001))))
            if len(points) < min_points:
                radius = min_dim * float(cfg.get("geometry_fallback_radius_fraction", 0.36))
                metrics.update(
                    {
                        "inner_candidate_count": 1.0,
                        "inner_geometry_fallback": 1.0,
                        "inner_x": prior_x,
                        "inner_y": prior_y,
                        "inner_radius": radius,
                        "inner_radius_fraction": radius / min_dim,
                        "inner_fill": 0.0,
                        "inner_stain_points": float(len(points)),
                    }
                )
                return image, (prior_x, prior_y, radius), metrics
            center_y, center_x = np.median(points, axis=0)
            distances = np.hypot(points[:, 1] - center_x, points[:, 0] - center_y)
            raw_radius_fraction = float(
                np.percentile(distances, float(cfg.get("stain_radius_percentile", 90.0)))
                / min_dim
            )
            radius_fraction = float(
                np.clip(
                    raw_radius_fraction,
                    float(cfg.get("min_inner_radius_fraction", 0.16)),
                    float(cfg.get("max_sparse_radius_fraction", 0.42)),
                )
            )
            radius = radius_fraction * min_dim
            center_distance = float(
                np.hypot(center_x - width / 2, center_y - height / 2) / min_dim
            )
            prior_distance = float(np.hypot(center_x - prior_x, center_y - prior_y) / min_dim)
            metrics.update(
                {
                    "inner_sparse_radius_fraction": radius_fraction,
                    "inner_sparse_raw_radius_fraction": raw_radius_fraction,
                    "inner_sparse_center_distance": center_distance,
                    "inner_sparse_prior_distance": prior_distance,
                    "inner_stain_points": float(len(points)),
                }
            )
            if not (
                float(cfg.get("min_inner_radius_fraction", 0.16))
                <= radius_fraction
                <= float(cfg.get("max_sparse_radius_fraction", 0.42))
                and center_distance <= float(cfg.get("max_inner_center_distance", 0.32))
                and prior_distance <= max_prior_distance
                and 0.50 / radius_fraction
                >= float(cfg.get("min_outer_to_inner_ratio", 1.12))
            ):
                radius = min_dim * float(cfg.get("geometry_fallback_radius_fraction", 0.36))
                metrics.update(
                    {
                        "inner_candidate_count": 1.0,
                        "inner_geometry_fallback": 1.0,
                        "inner_x": prior_x,
                        "inner_y": prior_y,
                        "inner_radius": radius,
                        "inner_radius_fraction": radius / min_dim,
                        "inner_fill": 0.0,
                    }
                )
                return image, (prior_x, prior_y, radius), metrics
            metrics.update(
                {
                    "inner_candidate_count": 1.0,
                    "inner_sparse_fallback": 1.0,
                    "inner_x": float(center_x),
                    "inner_y": float(center_y),
                    "inner_radius": radius,
                    "inner_radius_fraction": radius_fraction,
                    "inner_center_distance": center_distance,
                    "inner_fill": 0.0,
                }
            )
            return image, (float(center_x), float(center_y), radius), metrics
        (
            _area,
            x,
            y,
            radius,
            area_fraction,
            radius_fraction,
            aspect,
            fill,
            circularity,
            center_distance,
            prior_distance,
            enclosing_x,
            enclosing_y,
            enclosing_radius,
        ) = max(candidates)
        radius_fraction = float(
            np.clip(
                radius / min_dim,
                float(cfg.get("density_min_radius_fraction", 0.28)),
                float(cfg.get("density_max_radius_fraction", 0.36)),
            )
        )
        radius = radius_fraction * min_dim
        metrics.update(
            {
                "inner_x": x,
                "inner_y": y,
                "inner_radius": radius,
                "inner_area_fraction": area_fraction,
                "inner_radius_fraction": radius_fraction,
                "inner_aspect": aspect,
                "inner_fill": fill,
                "inner_circularity": circularity,
                "inner_center_distance": center_distance,
                "inner_prior_distance": prior_distance,
                "inner_enclosing_x": enclosing_x,
                "inner_enclosing_y": enclosing_y,
                "inner_enclosing_radius": enclosing_radius,
            }
        )
        return image, (x, y, radius), metrics

    def is_applicable(self, thumbnail: np.ndarray, config: Config | None = None):
        try:
            _image, inner, metrics = self._analyze(thumbnail, config)
        except Exception as exc:
            return decision(self.name, "error", f"Invalid thumbnail: {exc}")
        if inner is None:
            return decision(
                self.name,
                "not_applicable",
                "Serrated perimeter and centered inner-density gates were not both satisfied",
                metrics=metrics,
            )
        return decision(
            self.name,
            "accepted",
            "Serrated outer perimeter with distinct inner cell-rich circle found",
            confidence=0.88,
            metrics=metrics,
        )

    def detect(
        self,
        thumbnail: np.ndarray,
        slide_size: tuple[int, int] | None = None,
        config: Config | None = None,
    ):
        image, inner, metrics = self._analyze(thumbnail, config)
        if inner is None:
            return decision(
                self.name,
                "rejected",
                "Serrated-circle safety gates failed",
                metrics=metrics,
            )
        x, y, radius = inner
        expansion_key = "hough_expansion" if metrics.get("inner_hough") else "expansion"
        expansion = float((config or {}).get(expansion_key, 1.0))
        radius *= expansion
        if radius >= metrics["outer_radius"] * 0.90:
            return decision(
                self.name,
                "rejected",
                "Expanded inner ROI approaches the scanner perimeter",
                metrics=metrics,
            )
        confidence = min(
            0.98,
            0.55
            + 1.5 * min(0.12, metrics["outer_artifact_fraction"])
            + 0.15 * min(1.0, metrics["inner_fill"]),
        )
        circle = circle_mask(image.shape[:2], x, y, radius)
        height, width = image.shape[:2]
        min_dim = min(height, width)
        yy, xx = np.indices((height, width), dtype=np.float32)
        safe_radius = min_dim * float(
            (config or {}).get("safe_inner_radius_fraction", 0.445)
        )
        safe_interior = (
            np.hypot(xx - (width - 1) / 2, yy - (height - 1) / 2) <= safe_radius
        )

        # The fitted circle is only a guaranteed core. Recover sparse or
        # asymmetric cell fields beyond it with a permissive density pass,
        # while the hard safe-interior gate keeps the serrated scanner edge out.
        density_image = image.copy()
        density_image[~safe_interior] = 255
        density_result = DensityHeuristic().detect(
            density_image,
            slide_size,
            (config or {}).get("within_density", {}),
        )
        if density_result.status == "accepted" and density_result.region is not None:
            density_mask = density_result.region.mask & safe_interior
            dilation = max(
                1,
                int(
                    round(
                        min_dim
                        * float((config or {}).get("density_expansion_ratio", 0.01))
                    )
                ),
            )
            dilation = dilation * 2 + 1
            density_mask = cv2.dilate(
                density_mask.astype(np.uint8),
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (dilation, dilation),
                ),
            ) > 0
            mask = (circle | density_mask) & safe_interior
            metrics["density_extension_fraction"] = float(
                np.count_nonzero(density_mask & ~circle) / mask.size
            )
            metrics["density_mask_fraction"] = float(density_mask.mean())
        else:
            mask = circle & safe_interior
            metrics["density_extension_fallback_to_core"] = 1.0

        metrics["circle_mask_fraction"] = float(circle.mean())
        metrics["safe_inner_radius"] = float(safe_radius)
        metrics["final_mask_fraction"] = float(mask.mean())
        if not mask.any():
            return decision(
                self.name,
                "rejected",
                "Inner high-recall mask is empty",
                confidence=confidence,
                metrics=metrics,
            )
        return decision(
            self.name,
            "accepted",
            "Selected high-recall inner density while excluding the serrated scan perimeter",
            confidence=confidence,
            region=make_region(mask, image, slide_size, {"method": self.name}),
            metrics=metrics,
        )


def serrated_outer_circle(
    thumbnail: np.ndarray,
    slide_size: tuple[int, int] | None = None,
    config: Config | None = None,
):
    return SerratedOuterCircleHeuristic()(thumbnail, slide_size, config)
