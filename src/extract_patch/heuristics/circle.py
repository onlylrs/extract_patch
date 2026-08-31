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
    odd_size,
    prepare_thumbnail,
    register,
)


def _content_contours(image: np.ndarray, config: Config | None = None):
    cfg = config or {}
    min_dim = min(image.shape[:2])
    seed = foreground_seed(image, color_delta=float(cfg.get("color_delta", 8.0)))
    sigma = max(1.0, min_dim * float(cfg.get("density_sigma_ratio", 0.012)))
    density = cv2.GaussianBlur(seed.astype(np.float32), (0, 0), sigma)
    threshold = max(0.025, float(density.max()) * float(cfg.get("threshold_ratio", 0.16)))
    binary = (density >= threshold).astype(np.uint8)
    size = odd_size(min_dim * float(cfg.get("morphology_ratio", 0.018)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return binary, sorted(contours, key=cv2.contourArea, reverse=True)


def _shape_metrics(contour: np.ndarray, shape: tuple[int, int]) -> dict[str, float]:
    height, width = shape
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    (x, y), radius = cv2.minEnclosingCircle(contour)
    bx, by, bw, bh = cv2.boundingRect(contour)
    hull = cv2.convexHull(contour)
    return {
        "x": float(x),
        "y": float(y),
        "radius": float(radius),
        "area_fraction": area / (height * width),
        "circularity": 4.0 * np.pi * area / max(1e-6, perimeter * perimeter),
        "solidity": area / max(1e-6, float(cv2.contourArea(hull))),
        "fill_enclosing": area / max(1e-6, np.pi * radius * radius),
        "aspect": max(bw, bh) / max(1, min(bw, bh)),
        "center_distance": float(np.hypot(x - width / 2, y - height / 2) / min(height, width)),
        "bbox_x": float(bx),
        "bbox_y": float(by),
    }


@register
class SingleCircleHeuristic(Heuristic):
    name = "single_circle"

    def _candidate(self, thumbnail: np.ndarray, config: Config | None):
        image = prepare_thumbnail(thumbnail)
        height, width = image.shape[:2]
        cfg = config or {}
        aspect = width / height
        if not float(cfg.get("min_canvas_aspect", 0.72)) <= aspect <= float(
            cfg.get("max_canvas_aspect", 1.40)
        ):
            return image, None, {"canvas_aspect": aspect}
        _binary, contours = _content_contours(image, cfg)
        if not contours:
            return image, None, {"canvas_aspect": aspect}
        metrics = _shape_metrics(contours[0], (height, width))
        metrics["canvas_aspect"] = aspect
        valid = (
            float(cfg.get("min_area_fraction", 0.08))
            <= metrics["area_fraction"]
            <= float(cfg.get("max_area_fraction", 0.88))
            and metrics["circularity"] >= float(cfg.get("min_circularity", 0.66))
            and metrics["solidity"] >= float(cfg.get("min_solidity", 0.86))
            and metrics["fill_enclosing"] >= float(cfg.get("min_fill_enclosing", 0.60))
            and metrics["center_distance"] <= float(cfg.get("max_center_distance", 0.22))
        )
        return image, (metrics if valid else None), metrics

    def is_applicable(self, thumbnail: np.ndarray, config: Config | None = None):
        try:
            _image, candidate, metrics = self._candidate(thumbnail, config)
        except Exception as exc:
            return decision(self.name, "error", f"Invalid thumbnail: {exc}")
        if candidate is None:
            return decision(
                self.name,
                "not_applicable",
                "No centered, round single-region geometry was found",
                metrics=metrics,
            )
        return decision(
            self.name, "accepted", "Centered circular geometry found", confidence=0.8, metrics=metrics
        )

    def detect(self, thumbnail, slide_size=None, config: Config | None = None):
        image, candidate, metrics = self._candidate(thumbnail, config)
        if candidate is None:
            return decision(
                self.name, "rejected", "Circular candidate failed safety gates", metrics=metrics
            )
        expansion = float((config or {}).get("expansion", 1.04))
        mask = circle_mask(
            image.shape[:2],
            candidate["x"],
            candidate["y"],
            candidate["radius"] * expansion,
        )
        confidence = min(
            0.99,
            0.35
            + 0.30 * candidate["circularity"]
            + 0.20 * candidate["fill_enclosing"]
            + 0.15 * candidate["solidity"],
        )
        return decision(
            self.name,
            "accepted",
            "Selected one centered cell-rich circle",
            confidence=confidence,
            region=make_region(mask, image, slide_size, {"method": self.name}),
            metrics=metrics,
        )


@register
class DoubleCircleHeuristic(Heuristic):
    name = "double_circle"

    def _candidates(self, thumbnail: np.ndarray, config: Config | None):
        image = prepare_thumbnail(thumbnail)
        height, width = image.shape[:2]
        cfg = config or {}
        aspect = width / height
        base: dict[str, Any] = {"canvas_aspect": aspect}
        if aspect < float(cfg.get("min_canvas_aspect", 1.45)):
            return image, None, base
        _binary, contours = _content_contours(image, cfg)
        candidates = []
        for contour in contours:
            metrics = _shape_metrics(contour, (height, width))
            if (
                float(cfg.get("min_area_fraction", 0.025))
                <= metrics["area_fraction"]
                <= float(cfg.get("max_area_fraction", 0.32))
                and metrics["circularity"] >= float(cfg.get("min_circularity", 0.58))
                and metrics["fill_enclosing"] >= float(cfg.get("min_fill_enclosing", 0.50))
            ):
                candidates.append(metrics)
        if len(candidates) < 2:
            base["candidate_count"] = len(candidates)
            return image, None, base
        pair = sorted(candidates[:4], key=lambda item: item["x"])
        best = max(
            (
                (left, right)
                for index, left in enumerate(pair)
                for right in pair[index + 1 :]
            ),
            key=lambda item: abs(item[1]["x"] - item[0]["x"]),
        )
        left, right = best
        radius_ratio = min(left["radius"], right["radius"]) / max(
            left["radius"], right["radius"]
        )
        separation = np.hypot(left["x"] - right["x"], left["y"] - right["y"]) / max(
            left["radius"], right["radius"]
        )
        y_alignment = abs(left["y"] - right["y"]) / min(height, width)
        base.update(
            {
                "candidate_count": len(candidates),
                "radius_ratio": float(radius_ratio),
                "separation": float(separation),
                "y_alignment": float(y_alignment),
            }
        )
        valid = (
            radius_ratio >= float(cfg.get("min_radius_ratio", 0.55))
            and separation >= float(cfg.get("min_separation", 2.3))
            and y_alignment <= float(cfg.get("max_y_misalignment", 0.22))
        )
        return image, ((left, right) if valid else None), base

    def is_applicable(self, thumbnail, config: Config | None = None):
        try:
            _image, pair, metrics = self._candidates(thumbnail, config)
        except Exception as exc:
            return decision(self.name, "error", f"Invalid thumbnail: {exc}")
        if pair is None:
            return decision(
                self.name,
                "not_applicable",
                "No safe pair of separated circular regions was found",
                metrics=metrics,
            )
        return decision(self.name, "accepted", "Two circular regions found", confidence=0.85, metrics=metrics)

    def detect(self, thumbnail, slide_size=None, config: Config | None = None):
        image, pair, metrics = self._candidates(thumbnail, config)
        if pair is None:
            return decision(self.name, "rejected", "Double-circle safety gates failed", metrics=metrics)
        expansion = float((config or {}).get("expansion", 1.04))
        mask = np.zeros(image.shape[:2], dtype=bool)
        for item in pair:
            mask |= circle_mask(
                image.shape[:2], item["x"], item["y"], item["radius"] * expansion
            )
        confidence = min(0.98, 0.55 + 0.25 * metrics["radius_ratio"] + 0.1 * min(1.0, metrics["separation"] / 4))
        return decision(
            self.name,
            "accepted",
            "Selected two separated cell-rich circles",
            confidence=confidence,
            region=make_region(mask, image, slide_size, {"method": self.name}),
            metrics=metrics,
        )


@register
class SmearHeuristic(Heuristic):
    name = "smear"

    def _candidate(self, thumbnail, config: Config | None):
        image = prepare_thumbnail(thumbnail)
        cfg = config or {}
        _binary, contours = _content_contours(image, cfg)
        if not contours:
            return image, None, {}
        metrics = _shape_metrics(contours[0], image.shape[:2])
        valid = (
            float(cfg.get("min_area_fraction", 0.10))
            <= metrics["area_fraction"]
            <= float(cfg.get("max_area_fraction", 0.90))
            and metrics["aspect"] >= float(cfg.get("min_aspect", 1.45))
            and metrics["solidity"] >= float(cfg.get("min_solidity", 0.80))
            and metrics["circularity"] >= float(cfg.get("min_circularity", 0.35))
        )
        return image, (contours[0] if valid else None), metrics

    def is_applicable(self, thumbnail, config: Config | None = None):
        try:
            _image, contour, metrics = self._candidate(thumbnail, config)
        except Exception as exc:
            return decision(self.name, "error", f"Invalid thumbnail: {exc}")
        if contour is None:
            return decision(
                self.name,
                "not_applicable",
                "No dominant elongated smear geometry was found",
                metrics=metrics,
            )
        return decision(self.name, "accepted", "Elongated smear geometry found", confidence=0.8, metrics=metrics)

    def detect(self, thumbnail, slide_size=None, config: Config | None = None):
        image, contour, metrics = self._candidate(thumbnail, config)
        if contour is None:
            return decision(self.name, "rejected", "Smear candidate failed safety gates", metrics=metrics)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 1, thickness=cv2.FILLED)
        expansion = float((config or {}).get("expansion_ratio", 0.015))
        if expansion > 0:
            size = odd_size(min(image.shape[:2]) * expansion)
            mask = cv2.dilate(
                mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            )
        confidence = min(0.97, 0.45 + 0.18 * min(metrics["aspect"], 2.5) + 0.1 * metrics["solidity"])
        return decision(
            self.name,
            "accepted",
            "Selected the dominant elongated smear",
            confidence=confidence,
            region=make_region(mask, image, slide_size, {"method": self.name}),
            metrics=metrics,
        )


def single_circle(thumbnail, slide_size=None, config: Config | None = None):
    return SingleCircleHeuristic()(thumbnail, slide_size, config)


def double_circle(thumbnail, slide_size=None, config: Config | None = None):
    return DoubleCircleHeuristic()(thumbnail, slide_size, config)


def smear(thumbnail, slide_size=None, config: Config | None = None):
    return SmearHeuristic()(thumbnail, slide_size, config)
