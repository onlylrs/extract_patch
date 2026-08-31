# ruff: noqa
"""Feature maps and double-circle detection for wide WSI thumbnails."""

import cv2
import numpy as np


def normalize(feature, low=2, high=99.5):
    lower, upper = np.percentile(feature, [low, high])
    normalized = np.clip((feature - lower) / max(1e-6, upper - lower), 0, 1)
    return (normalized * 255).astype(np.uint8)


def feature_maps(rgb):
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
    texture_square = cv2.GaussianBlur(texture_source * texture_source, (0, 0), texture_sigma)
    local_texture = np.sqrt(np.maximum(0.0, texture_square - texture_mean * texture_mean))
    texture_density = cv2.GaussianBlur(local_texture, (0, 0), max(4.0, min_dim * 0.025))
    hsv_stain = saturation * np.clip((gray.astype(np.float32) - 35) / 120, 0, 1)
    lab_chroma = chroma * np.clip((gray.astype(np.float32) - 35) / 120, 0, 1)
    background = cv2.GaussianBlur(lab_chroma, (0, 0), max(8.0, min_dim * 0.09))
    local_chroma = np.maximum(lab_chroma - background * 0.45, 0)
    density = cv2.GaussianBlur((lab_chroma >= 6).astype(np.float32), (0, 0), max(3.0, min_dim * 0.018))
    density = np.maximum(density - cv2.GaussianBlur(neutral_dark, (0, 0), max(3.0, min_dim * 0.012)) * 0.4, 0)
    return {
        "texture_density": normalize(texture_density, 2, 99),
        "stain_density": normalize(density, 1, 99),
        "local_chroma": normalize(local_chroma),
    }


def detect_two(feature):
    height, width = feature.shape
    min_dim = min(height, width)
    blur = cv2.GaussianBlur(feature, (0, 0), max(2.0, min_dim * 0.012))
    edge = cv2.Canny(blur, 25, 70)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(40, int(width * 0.42)),
        param1=70,
        param2=max(12, int(min_dim * 0.018)),
        minRadius=int(min_dim * 0.18),
        maxRadius=int(min_dim * 0.39),
    )
    candidates = [] if circles is None else np.round(circles[0]).astype(int).tolist()
    selected = []
    for left, right in ((0, width // 2), (width // 2, width)):
        side = [circle for circle in candidates if left <= circle[0] < right]
        if not side:
            selected.append(None)
            continue
        scored = []
        yy, xx = np.ogrid[:height, :width]
        for x, y, radius in side:
            distance = np.hypot(xx - x, yy - y)
            inside = distance <= radius * 0.82
            outside = (distance >= radius * 1.05) & (distance <= radius * 1.25)
            contrast = float(feature[inside].mean() - feature[outside].mean()) if outside.any() else 0
            ring_support = float((edge[(distance >= radius * 0.94) & (distance <= radius * 1.06)] > 0).mean())
            center_penalty = abs(x - (left + right) / 2) / max(1, right - left)
            scored.append((contrast + 80 * ring_support - 30 * center_penalty, [x, y, radius]))
        selected.append(max(scored, key=lambda item: item[0])[1])
    return candidates, selected


def detect_texture_blobs(feature):
    height, width = feature.shape
    min_dim = min(height, width)
    selected = []
    for left, right in ((0, width // 2), (width // 2, width)):
        side = feature[:, left:right]
        threshold, _binary = cv2.threshold(side, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = (side >= max(threshold, np.percentile(side, 68))).astype(np.uint8) * 255
        kernel_size = max(5, int(round(min_dim * 0.025)) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        scored = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < height * width * 0.003:
                continue
            (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
            circle_area = np.pi * radius * radius
            compactness = area / max(1.0, circle_area)
            if compactness < 0.32 or not min_dim * 0.14 <= radius <= min_dim * 0.39:
                continue
            scored.append(
                (
                    area * compactness,
                    [int(round(center_x + left)), int(round(center_y)), int(round(radius * 1.04))],
                )
            )
        selected.append(max(scored, key=lambda item: item[0])[1] if scored else None)
    return selected
