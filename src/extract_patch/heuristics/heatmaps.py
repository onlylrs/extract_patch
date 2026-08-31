from __future__ import annotations

import cv2
import numpy as np


def normalize_heatmap(feature: np.ndarray, low: float = 2, high: float = 99.5) -> np.ndarray:
    lower, upper = np.percentile(feature, [low, high])
    normalized = np.clip((feature - lower) / max(1e-6, upper - lower), 0, 1)
    return (normalized * 255).astype(np.uint8)


def circle_feature_maps(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Build SmartCyto-style texture/stain maps while suppressing neutral rings.

    Neutral dark rings include common bubble/foam boundaries. Inpainting them
    before texture estimation prevents those boundaries from winning circle
    detection over the stained carrier.
    """

    height, width = rgb.shape[:2]
    min_dim = min(height, width)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    chroma = np.hypot(lab[:, :, 1] - 128, lab[:, :, 2] - 128)
    saturation = hsv[:, :, 1].astype(np.float32)
    neutral_dark = ((gray < 120) & (chroma < 7)).astype(np.float32)

    neutral_ring = ((gray < 135) & (chroma < 8)).astype(np.uint8) * 255
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    neutral_ring = cv2.dilate(neutral_ring, ring_kernel)
    ring_suppressed = cv2.inpaint(gray, neutral_ring, 5, cv2.INPAINT_TELEA)

    texture_source = ring_suppressed.astype(np.float32)
    texture_sigma = max(2.0, min_dim * 0.008)
    texture_mean = cv2.GaussianBlur(texture_source, (0, 0), texture_sigma)
    texture_square = cv2.GaussianBlur(
        texture_source * texture_source, (0, 0), texture_sigma
    )
    local_texture = np.sqrt(
        np.maximum(0.0, texture_square - texture_mean * texture_mean)
    )
    texture_density = cv2.GaussianBlur(
        local_texture, (0, 0), max(4.0, min_dim * 0.025)
    )

    brightness_gate = np.clip((gray.astype(np.float32) - 35) / 120, 0, 1)
    lab_chroma = chroma * brightness_gate
    background = cv2.GaussianBlur(
        lab_chroma, (0, 0), max(8.0, min_dim * 0.09)
    )
    local_chroma = np.maximum(lab_chroma - background * 0.45, 0)
    density = cv2.GaussianBlur(
        (lab_chroma >= 6).astype(np.float32),
        (0, 0),
        max(3.0, min_dim * 0.018),
    )
    density = np.maximum(
        density
        - cv2.GaussianBlur(
            neutral_dark, (0, 0), max(3.0, min_dim * 0.012)
        )
        * 0.4,
        0,
    )
    return {
        "texture_density": normalize_heatmap(texture_density, 2, 99),
        "stain_density": normalize_heatmap(density, 1, 99),
        "local_chroma": normalize_heatmap(local_chroma),
        "saturation": normalize_heatmap(saturation),
    }


def circle_detection_heatmap(rgb: np.ndarray) -> np.ndarray:
    maps = circle_feature_maps(rgb)
    # Local chroma reveals the circular stain boundary most directly. Texture
    # and stain-density maps remain available to other heuristic plugins.
    return maps["local_chroma"]
