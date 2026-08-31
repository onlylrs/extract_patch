from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .base import (
    Config,
    Heuristic,
    decision,
    foreground_seed,
    make_region,
    odd_size,
    prepare_thumbnail,
    register,
)


@register
class DensityHeuristic(Heuristic):
    """General-purpose cell-density segmentation and universal fallback."""

    name = "density"

    def is_applicable(
        self, thumbnail: np.ndarray, config: Config | None = None
    ):
        try:
            image = prepare_thumbnail(thumbnail)
        except Exception as exc:
            return decision(self.name, "error", f"Invalid thumbnail: {exc}")
        return decision(
            self.name,
            "accepted",
            "Density segmentation is applicable to every valid thumbnail",
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
        cfg: dict[str, Any] = {
            "color_delta": 9.0,
            "density_sigma_ratio": 0.018,
            "density_threshold_ratio": 0.12,
            "close_ratio": 0.025,
            "open_ratio": 0.008,
            "min_component_area_ratio": 0.0005,
            "max_mask_fraction": 0.95,
        }
        cfg.update(config or {})
        height, width = image.shape[:2]
        min_dim = min(height, width)
        seed = foreground_seed(image, color_delta=float(cfg["color_delta"])).astype(np.float32)
        seed_fraction = float(seed.mean())
        if not seed.any():
            return decision(
                self.name,
                "rejected",
                "No foreground evidence was found",
                metrics={"seed_fraction": 0.0},
            )

        sigma = max(1.2, min_dim * float(cfg["density_sigma_ratio"]))
        density_map = cv2.GaussianBlur(seed, (0, 0), sigmaX=sigma, sigmaY=sigma)
        maximum = float(density_map.max())
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (odd_size(min_dim * float(cfg["close_ratio"])),) * 2
        )
        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (odd_size(min_dim * float(cfg["open_ratio"])),) * 2
        )
        min_area = max(8.0, height * width * float(cfg["min_component_area_ratio"]))
        best = np.zeros((height, width), dtype=np.uint8)
        used_cutoff = 0.0
        for multiplier in (1.0, 1.5, 2.25, 3.25):
            used_cutoff = max(
                0.008, maximum * float(cfg["density_threshold_ratio"]) * multiplier
            )
            binary = (density_map >= used_cutoff).astype(np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            kept = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
            if not kept and contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) >= 8:
                    kept = [largest]
            best.fill(0)
            if kept:
                cv2.drawContours(best, kept, -1, 1, thickness=cv2.FILLED)
            if 0 < float(best.mean()) <= float(cfg["max_mask_fraction"]):
                break

        fraction = float(best.mean())
        metrics = {
            "seed_fraction": seed_fraction,
            "mask_fraction": fraction,
            "max_density": maximum,
            "density_cutoff": used_cutoff,
        }
        if not best.any():
            return decision(
                self.name,
                "rejected",
                "Foreground evidence did not form a safe connected region",
                metrics=metrics,
            )
        if fraction > float(cfg["max_mask_fraction"]):
            return decision(
                self.name,
                "rejected",
                "Density mask covers too much of the thumbnail",
                metrics=metrics,
            )
        confidence = min(0.98, 0.45 + 2.0 * min(fraction, 0.25) + min(seed_fraction, 0.1))
        return decision(
            self.name,
            "accepted",
            "Foreground density produced a non-empty region",
            confidence=confidence,
            region=make_region(best, image, slide_size, {"method": self.name}),
            metrics=metrics,
        )


def density(
    thumbnail: np.ndarray,
    slide_size: tuple[int, int] | None = None,
    config: Config | None = None,
):
    return DensityHeuristic()(thumbnail, slide_size, config)
