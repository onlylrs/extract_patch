from __future__ import annotations

import numpy as np

from .base import Config, Heuristic, decision, make_region, prepare_thumbnail, register
from .density import DensityHeuristic
from .smartcyto_circle import (
    candidate_metrics,
    circle_candidates,
    circle_roi_has_content_contrast,
    feature_edge,
    make_roi,
    refine_single_pad_with_enclosing,
    resize_for_detection,
    select_circles_with_cascade,
    single_roi_is_safe,
)


@register
class SmartCytoCircleHeuristic(Heuristic):
    """The complete SmartCyto single/double/texture/stain/radial cascade."""

    name = "smartcyto_circle"

    def is_applicable(self, thumbnail: np.ndarray, config: Config | None = None):
        try:
            image = prepare_thumbnail(thumbnail)
        except Exception as exc:
            return decision(self.name, "error", f"Invalid thumbnail: {exc}")
        return decision(
            self.name,
            "accepted",
            "SmartCyto circle cascade can inspect this thumbnail",
            confidence=1.0,
            metrics={"height": image.shape[0], "width": image.shape[1]},
        )

    def detect(
        self,
        thumbnail: np.ndarray,
        slide_size: tuple[int, int] | None = None,
        config: Config | None = None,
    ):
        image = prepare_thumbnail(thumbnail)
        cfg = config or {}
        detection, scale = resize_for_detection(
            image, int(cfg.get("max_detection_dimension", 1200))
        )
        edge = feature_edge(detection)
        raw = circle_candidates(detection, edge)
        candidates = [candidate_metrics(detection, edge, circle) for circle in raw]
        mode, circles = select_circles_with_cascade(
            candidates,
            detection,
            edge=edge,
            pad_aspect_min=float(cfg.get("pad_aspect_min", 0.75)),
            pad_aspect_max=float(cfg.get("pad_aspect_max", 1.35)),
            wide_aspect_min=float(cfg.get("wide_aspect_min", 1.65)),
        )
        if not circles:
            return decision(
                self.name,
                "not_applicable",
                "SmartCyto circle cascade found no safe ROI",
                metrics={"mode": mode, "candidate_count": len(candidates)},
            )
        roi = make_roi(
            image.shape[:2],
            circles,
            scale,
            expansion=float(cfg.get("expansion", 1.01)),
        ).astype(np.uint8)
        density = DensityHeuristic().detect(
            image, slide_size, cfg.get("within_density", {})
        )
        if density.status == "accepted" and density.region is not None:
            base_mask = density.region.mask.astype(np.uint8) * 255
            if mode.startswith(("single_texture_", "double_")) and not (
                circle_roi_has_content_contrast(image, base_mask, roi, mode)
            ):
                return decision(
                    self.name,
                    "rejected",
                    f"SmartCyto mode={mode} failed carrier contrast safety gate",
                    metrics={"mode": mode, "candidate_count": len(candidates)},
                )
            if mode == "single" and not single_roi_is_safe(image, base_mask, roi):
                return decision(
                    self.name,
                    "rejected",
                    "SmartCyto single circle failed retained-content safety gate",
                    metrics={"mode": mode, "candidate_count": len(candidates)},
                )
            constrained = ((base_mask > 0).astype(np.uint8) & roi).astype(np.uint8) * 255
            constrained, mode = refine_single_pad_with_enclosing(
                image,
                base_mask,
                constrained,
                mode,
                expansion=float(cfg.get("enclosing_expansion", 1.0)),
            )
            mask = constrained > 0
        else:
            mask = roi > 0
        confidence = float(
            np.clip(
                np.mean([circle.get("score", 0.75) for circle in circles]),
                0.0,
                1.0,
            )
        )
        return decision(
            self.name,
            "accepted",
            f"SmartCyto circle cascade selected mode={mode}",
            confidence=confidence,
            region=make_region(mask, image, slide_size, {"method": self.name, "mode": mode}),
            metrics={
                "mode": mode,
                "candidate_count": len(candidates),
                "circle_count": len(circles),
            },
        )


def smartcyto_circle(
    thumbnail: np.ndarray,
    slide_size: tuple[int, int] | None = None,
    config: Config | None = None,
):
    return SmartCytoCircleHeuristic()(thumbnail, slide_size, config)
