# ruff: noqa
"""Complete SmartCyto circle-prior cascade, vendored as pure image algorithms."""

from typing import Optional, Tuple

import cv2
import numpy as np

from .smartcyto_preprocess import detect_texture_blobs, detect_two, feature_maps


def resize_for_detection(rgb, max_dimension=1200):
    scale = min(1.0, max_dimension / max(rgb.shape[:2]))
    if scale == 1.0:
        return rgb, scale
    return cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), scale


def feature_edge(rgb):
    height, width = rgb.shape[:2]
    min_dim = min(height, width)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sigma = max(3.0, min_dim * 0.018)
    mean = cv2.GaussianBlur(gray, (0, 0), sigma)
    mean_square = cv2.GaussianBlur(gray * gray, (0, 0), sigma)
    local_std = np.sqrt(np.maximum(0.0, mean_square - mean * mean))
    features = [local_std]
    features.extend(
        cv2.GaussianBlur(lab[:, :, channel], (0, 0), max(4.0, min_dim * 0.025))
        for channel in range(3)
    )
    edge = np.zeros((height, width), dtype=np.float32)
    for feature in features:
        low, high = np.percentile(feature, [2, 98])
        normalized = (feature - low) / max(1e-6, high - low)
        grad_x = cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(normalized, cv2.CV_32F, 0, 1, ksize=3)
        edge += np.hypot(grad_x, grad_y)
    edge = cv2.GaussianBlur(edge, (0, 0), max(1.0, min_dim * 0.006))
    return edge / max(1e-6, np.percentile(edge, 99.5))


def estimate_pad_bounds(rgb):
    height, width = rgb.shape[:2]
    min_dim = min(height, width)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sigma = max(1.5, min_dim * 0.006)
    mean = cv2.GaussianBlur(gray, (0, 0), sigma)
    mean_square = cv2.GaussianBlur(gray * gray, (0, 0), sigma)
    texture = np.sqrt(np.maximum(mean_square - mean * mean, 0))
    density = cv2.GaussianBlur(texture, (0, 0), max(4.0, min_dim * 0.02))
    threshold = max(float(np.percentile(density, 55)), 2.0)
    mask = (density >= threshold).astype(np.uint8) * 255
    kernel_size = max(7, int(round(min_dim * 0.035)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, candidate_width, candidate_height = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        fill = area / max(1, candidate_width * candidate_height)
        if area >= height * width * 0.08 and fill >= 0.35:
            candidates.append((area * fill, (x, y, candidate_width, candidate_height)))
    if not candidates:
        return 0, 0, width, height
    return max(candidates, key=lambda item: item[0])[1]


def pad_bounds_texture(rgb, bounds):
    x, y, width, height = bounds
    min_dim = min(rgb.shape[:2])
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sigma = max(1.5, min_dim * 0.006)
    mean = cv2.GaussianBlur(gray, (0, 0), sigma)
    mean_square = cv2.GaussianBlur(gray * gray, (0, 0), sigma)
    texture = np.sqrt(np.maximum(mean_square - mean * mean, 0))
    inset = max(1, int(round(min(width, height) * 0.05)))
    crop = texture[y + inset:y + height - inset, x + inset:x + width - inset]
    return float(np.median(crop)) if crop.size else 0.0


def carrier_texture_metrics(rgb, circle):
    height, width = rgb.shape[:2]
    min_dim = min(height, width)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sigma = max(1.5, min_dim * 0.006)
    mean = cv2.GaussianBlur(gray, (0, 0), sigma)
    mean_square = cv2.GaussianBlur(gray * gray, (0, 0), sigma)
    texture = np.sqrt(np.maximum(mean_square - mean * mean, 0))
    texture = cv2.GaussianBlur(texture, (0, 0), max(2.0, min_dim * 0.008))
    yy, xx = np.ogrid[:height, :width]
    distance = np.hypot(xx - circle["x"], yy - circle["y"])
    inner = distance <= circle["radius"] * 0.78
    annulus = (distance >= circle["radius"] * 1.16) & (distance <= circle["radius"] * 1.75)
    inner_median = float(np.median(texture[inner])) if inner.any() else 0.0
    annulus_median = float(np.median(texture[annulus])) if annulus.any() else 0.0
    return {
        "inner_median": inner_median,
        "annulus_median": annulus_median,
        "texture_ratio": annulus_median / max(0.25, inner_median),
    }


def ring_support(edge, x, y, radius, angles):
    height, width = edge.shape
    px = np.rint(x + np.cos(angles) * radius).astype(np.int32)
    py = np.rint(y + np.sin(angles) * radius).astype(np.int32)
    if np.any(px < 0) or np.any(py < 0) or np.any(px >= width) or np.any(py >= height):
        return None
    values = edge[py, px]
    return (
        float(np.mean(values >= 0.18)),
        float(np.mean(values)),
        float(np.percentile(values, 25)),
    )


def detect_pad_radial_circle(rgb, edge, bounds):
    x0, y0, width, height = bounds
    pad_min = min(width, height)
    center_x = x0 + width / 2
    center_y = y0 + height / 2
    angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    center_extent = pad_min * 0.16
    center_step = max(5, int(round(pad_min * 0.018)))
    radius_step = max(3, int(round(pad_min * 0.009)))
    ranked = []
    dilated_edges = {}
    for y in range(int(center_y - center_extent), int(center_y + center_extent) + 1, center_step):
        for x in range(int(center_x - center_extent), int(center_x + center_extent) + 1, center_step):
            center_distance = np.hypot(x - center_x, y - center_y) / pad_min
            for radius in range(int(pad_min * 0.16), int(pad_min * 0.48) + 1, radius_step):
                window = max(2, int(round(radius * 0.025)))
                if window not in dilated_edges:
                    kernel = np.ones((window * 2 + 1, window * 2 + 1), dtype=np.uint8)
                    dilated_edges[window] = cv2.dilate(edge, kernel)
                support = ring_support(dilated_edges[window], x, y, radius, angles)
                if support is None:
                    continue
                coverage, strength, lower_quartile = support
                if coverage < 0.72 or lower_quartile < 0.12:
                    continue
                radius_ratio = radius / pad_min
                score = (
                    1.35 * coverage
                    + strength
                    + 0.55 * lower_quartile
                    - 1.1 * center_distance
                    + 0.55 * radius_ratio
                )
                ranked.append(
                    (score, [x, y, radius], coverage, lower_quartile, center_distance, radius_ratio)
                )
    if not ranked:
        return None
    ranked.sort(reverse=True)
    best_score = ranked[0][0]
    competitive = [
        item
        for item in ranked
        if item[0] >= best_score - 0.12
        and item[2] >= 0.72
        and item[3] >= 0.12
        and item[4] <= 0.16
    ]
    if not competitive:
        return ranked[0][1]
    return max(competitive, key=lambda item: (item[5], item[0]))[1]


def select_pad_hough_in_bounds(candidates, bounds):
    x0, y0, width, height = bounds
    pad_min = min(width, height)
    center_x = x0 + width / 2
    center_y = y0 + height / 2
    accepted = []
    for candidate in candidates:
        relative_center = np.hypot(candidate["x"] - center_x, candidate["y"] - center_y) / pad_min
        relative_radius = candidate["radius"] / pad_min
        if (
            relative_center <= 0.28
            and 0.18 <= relative_radius <= 0.49
            and candidate["visible_fraction"] >= 0.65
            and candidate["boundary_coverage"] >= 0.48
            and (candidate["color_delta"] >= 4.0 or candidate["texture_delta"] >= 5.0)
        ):
            accepted.append((candidate, relative_center, relative_radius))
    if not accepted:
        return None
    best_quality = max(
        candidate["boundary_coverage"] + min(1.0, candidate["boundary_strength"])
        for candidate, _center, _radius in accepted
    )
    competitive = [
        item
        for item in accepted
        if item[0]["boundary_coverage"] + min(1.0, item[0]["boundary_strength"])
        >= best_quality - 0.22
    ]
    return max(
        competitive,
        key=lambda item: (-item[1], item[0]["boundary_coverage"], item[2]),
    )[0]


def select_texture_pad_circle(candidates, rgb, edge, min_carrier_texture=14.0):
    bounds = estimate_pad_bounds(rgb)
    selected = select_pad_hough_in_bounds(candidates, bounds)
    mode = "single_texture_hough"
    if selected is None and pad_bounds_texture(rgb, bounds) >= min_carrier_texture:
        radial_circle = detect_pad_radial_circle(rgb, edge, bounds)
        if radial_circle is not None:
            selected = candidate_metrics(rgb, edge, radial_circle)
            mode = "single_texture_radial"
    if selected is None:
        return None, "fallback"
    carrier = carrier_texture_metrics(rgb, selected)
    if carrier["annulus_median"] < min_carrier_texture:
        return None, "fallback"
    return selected, mode


def circle_candidates(rgb, edge):
    height, width = rgb.shape[:2]
    min_dim = min(height, width)
    edge_u8 = np.clip(edge * 255, 0, 255).astype(np.uint8)
    circles = cv2.HoughCircles(
        edge_u8,
        cv2.HOUGH_GRADIENT,
        dp=1.3,
        minDist=max(35, int(min_dim * 0.18)),
        param1=60,
        param2=max(15, int(min_dim * 0.019)),
        minRadius=max(18, int(min_dim * 0.08)),
        maxRadius=int(min_dim * 0.49),
    )
    return [] if circles is None else np.round(circles[0]).astype(int).tolist()


def candidate_metrics(rgb, edge, circle):
    x, y, radius = circle
    height, width = rgb.shape[:2]
    min_dim = min(height, width)
    yy, xx = np.ogrid[:height, :width]
    distance = np.hypot(xx - x, yy - y)
    inner = distance <= radius * 0.82
    outer = (distance >= radius * 1.06) & (distance <= radius * 1.24)
    visible_disk = distance <= radius
    disk_area = np.pi * radius * radius
    visible_fraction = min(1.0, float(visible_disk.sum() / max(1.0, disk_area)))

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    chroma = np.hypot(lab[:, :, 1] - 128, lab[:, :, 2] - 128)
    stain = chroma >= 8.0

    angle_support = []
    for angle in np.linspace(0, 2 * np.pi, 48, endpoint=False):
        px = int(round(x + np.cos(angle) * radius))
        py = int(round(y + np.sin(angle) * radius))
        if px < 0 or py < 0 or px >= width or py >= height:
            continue
        window = max(2, int(round(radius * 0.025)))
        crop = edge[max(0, py - window):min(height, py + window + 1), max(0, px - window):min(width, px + window + 1)]
        angle_support.append(float(crop.max()) if crop.size else 0.0)
    boundary_coverage = float(np.mean(np.asarray(angle_support) >= 0.18)) if angle_support else 0.0
    boundary_strength = float(np.mean(np.asarray(angle_support))) if angle_support else 0.0

    inner_std = float(gray[inner].std()) if inner.any() else 0.0
    outer_std = float(gray[outer].std()) if outer.any() else inner_std
    color_delta = 0.0
    if inner.any() and outer.any():
        color_delta = float(np.linalg.norm(lab[inner].mean(axis=0) - lab[outer].mean(axis=0)))
    texture_delta = abs(inner_std - outer_std)

    spread_cells = []
    for row in range(6):
        for column in range(6):
            left = x - radius * 0.82 + column * radius * 1.64 / 6
            right = x - radius * 0.82 + (column + 1) * radius * 1.64 / 6
            top = y - radius * 0.82 + row * radius * 1.64 / 6
            bottom = y - radius * 0.82 + (row + 1) * radius * 1.64 / 6
            cell = inner & (xx >= left) & (xx < right) & (yy >= top) & (yy < bottom)
            if cell.sum() >= 20:
                spread_cells.append(float(stain[cell].mean()))
    content_spread = float(np.mean(np.asarray(spread_cells) >= 0.008)) if spread_cells else 0.0
    stain_ratio = float(stain[inner].mean()) if inner.any() else 0.0

    center_distance = np.hypot(x - width / 2, y - height / 2) / max(1.0, min_dim)
    radius_ratio = radius / max(1.0, min_dim)
    contrast_score = min(1.0, color_delta / 18.0 + texture_delta / 25.0)
    content_score = min(1.0, content_spread / 0.45 + stain_ratio / 0.05) / 2
    boundary_score = min(1.0, boundary_coverage / 0.65 + boundary_strength / 0.35) / 2
    score = 0.42 * boundary_score + 0.32 * contrast_score + 0.26 * content_score

    return {
        "x": int(x),
        "y": int(y),
        "radius": int(radius),
        "radius_ratio": radius_ratio,
        "center_distance": center_distance,
        "visible_fraction": visible_fraction,
        "boundary_coverage": boundary_coverage,
        "boundary_strength": boundary_strength,
        "color_delta": color_delta,
        "texture_delta": texture_delta,
        "content_spread": content_spread,
        "stain_ratio": stain_ratio,
        "score": score,
    }


def stain_centroid(rgb, left, right):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    chroma = np.hypot(lab[:, :, 1] - 128, lab[:, :, 2] - 128)
    weights = np.maximum(chroma - 8.0, 0.0)
    weights[:, :left] = 0
    weights[:, right:] = 0
    threshold = np.percentile(weights[weights > 0], 60) if np.any(weights > 0) else 0
    weights[weights < threshold] = 0
    total = float(weights.sum())
    if total <= 0:
        return ((left + right) / 2, rgb.shape[0] / 2)
    yy, xx = np.indices(weights.shape)
    return (float((xx * weights).sum() / total), float((yy * weights).sum() / total))


def fit_content_circle(rgb, left, right):
    height, width = rgb.shape[:2]
    min_dim = min(height, width)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    chroma = np.hypot(lab[:, :, 1] - 128, lab[:, :, 2] - 128)
    seed = (chroma >= 8.0).astype(np.float32)
    seed[:, :left] = 0
    seed[:, right:] = 0
    density = cv2.GaussianBlur(seed, (0, 0), max(3.0, min_dim * 0.025))
    mask = (density >= 0.003).astype(np.uint8)
    kernel_size = max(5, int(round(min_dim * 0.02)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask[:, :left] = 0
    mask[:, right:] = 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) >= height * width * 0.001]
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
    radius = float(np.clip(radius * 1.04, min_dim * 0.16, min_dim * 0.38))
    return {
        "x": int(round(center_x)),
        "y": int(round(center_y)),
        "radius": int(round(radius)),
        "radius_ratio": radius / min_dim,
        "center_distance": np.hypot(center_x - width / 2, center_y - height / 2) / min_dim,
        "visible_fraction": 1.0,
        "boundary_coverage": 0.0,
        "boundary_strength": 0.0,
        "color_delta": 0.0,
        "texture_delta": 0.0,
        "content_spread": 0.0,
        "stain_ratio": float(seed[mask > 0].mean()) if np.any(mask > 0) else 0.0,
        "score": 1.0,
    }


def raw_to_circles(raw_selected, rgb):
    height, width = rgb.shape[:2]
    min_dim = min(height, width)
    circles = []
    for item in raw_selected:
        if item is None:
            return None
        x, y, radius = item
        circles.append(
            {
                "x": int(x),
                "y": int(y),
                "radius": int(radius),
                "radius_ratio": radius / min_dim,
                "center_distance": np.hypot(x - width / 2, y - height / 2) / min_dim,
                "visible_fraction": 1.0,
                "boundary_coverage": 0.0,
                "boundary_strength": 0.0,
                "color_delta": 0.0,
                "texture_delta": 0.0,
                "content_spread": 0.0,
                "stain_ratio": 0.0,
                "score": 1.0,
            }
        )
    return circles


def validate_double_circles(left, right, min_separation=2.6, min_radius_ratio=0.5):
    if left is None or right is None:
        return False
    radius_ratio = min(left["radius"], right["radius"]) / max(left["radius"], right["radius"])
    distance = np.hypot(left["x"] - right["x"], left["y"] - right["y"])
    return radius_ratio >= min_radius_ratio and distance >= max(left["radius"], right["radius"]) * min_separation


def select_pad_circle(candidates, rgb, pad_aspect_min=0.75, pad_aspect_max=1.35):
    height, width = rgb.shape[:2]
    aspect_ratio = width / max(1, height)
    if not (pad_aspect_min <= aspect_ratio <= pad_aspect_max):
        return None
    central = [
        candidate
        for candidate in candidates
        if candidate["center_distance"] <= 0.13
        and candidate["radius_ratio"] >= 0.18
        and candidate["visible_fraction"] >= 0.65
        and candidate["boundary_coverage"] >= 0.48
        and (candidate["color_delta"] >= 4.0 or candidate["texture_delta"] >= 5.0)
    ]
    if not central:
        return None
    return max(central, key=lambda item: item["radius"] * (0.7 + item["boundary_coverage"]))


def select_double_edge(candidates, rgb):
    height, width = rgb.shape[:2]
    min_dim = min(height, width)

    content_left = fit_content_circle(rgb, 0, width // 2)
    content_right = fit_content_circle(rgb, width // 2, width)
    if content_left is not None and content_right is not None:
        for content, is_left in ((content_left, True), (content_right, False)):
            nearby = [
                candidate
                for candidate in candidates
                if (candidate["x"] < width / 2) == is_left
                and 0.16 <= candidate["radius_ratio"] <= 0.38
                and np.hypot(candidate["x"] - content["x"], candidate["y"] - content["y"]) <= min_dim * 0.28
            ]
            if nearby:
                scale_candidate = min(
                    nearby,
                    key=lambda item: np.hypot(item["x"] - content["x"], item["y"] - content["y"]),
                )
                content["radius"] = max(content["radius"], scale_candidate["radius"])
                content["radius_ratio"] = content["radius"] / min_dim
        if validate_double_circles(content_left, content_right):
            return [content_left, content_right]

    left_center = stain_centroid(rgb, 0, width // 2)
    right_center = stain_centroid(rgb, width // 2, width)

    def side_best(side, target):
        if side == "left":
            side_candidates = [candidate for candidate in candidates if candidate["x"] < width * 0.43]
        else:
            side_candidates = [candidate for candidate in candidates if candidate["x"] > width * 0.57]
        side_candidates = [
            candidate
            for candidate in side_candidates
            if 0.16 <= candidate["radius_ratio"] <= 0.38
            and candidate["visible_fraction"] >= 0.55
            and candidate["boundary_coverage"] >= 0.30
        ]
        if not side_candidates:
            return None
        return max(
            side_candidates,
            key=lambda item: (
                item["boundary_coverage"]
                + min(0.5, item["content_spread"])
                - 1.4 * np.hypot(item["x"] - target[0], item["y"] - target[1]) / min_dim
                - 0.5 * abs(item["radius_ratio"] - 0.29)
            ),
        )

    left = side_best("left", left_center)
    right = side_best("right", right_center)
    if left is not None and right is not None:
        radius_ratio = min(left["radius"], right["radius"]) / max(left["radius"], right["radius"])
        distance = np.hypot(left["x"] - right["x"], left["y"] - right["y"])
        if radius_ratio >= 0.55 and distance >= max(left["radius"], right["radius"]) * 3.5:
            return [left, right]
    return None


def select_circles_with_cascade(
    candidates,
    detection_rgb,
    *,
    edge=None,
    pad_aspect_min=0.75,
    pad_aspect_max=1.35,
    wide_aspect_min=1.65,
):
    height, width = detection_rgb.shape[:2]
    aspect_ratio = width / max(1, height)

    if pad_aspect_min <= aspect_ratio <= pad_aspect_max:
        pad_circle = select_pad_circle(candidates, detection_rgb, pad_aspect_min, pad_aspect_max)
        if pad_circle is not None:
            return "single", [pad_circle]

    if aspect_ratio >= wide_aspect_min:
        maps = feature_maps(detection_rgb)

        texture_raw = detect_texture_blobs(maps["texture_density"])
        texture_circles = raw_to_circles(texture_raw, detection_rgb)
        if texture_circles and validate_double_circles(texture_circles[0], texture_circles[1]):
            return "double_texture", texture_circles

        _stain_candidates, stain_raw = detect_two(maps["stain_density"])
        stain_circles = raw_to_circles(stain_raw, detection_rgb)
        if stain_circles and validate_double_circles(stain_circles[0], stain_circles[1]):
            return "double_stain", stain_circles

        edge_circles = select_double_edge(candidates, detection_rgb)
        if edge_circles is not None:
            return "double_edge", edge_circles

    texture_pad, texture_pad_mode = select_texture_pad_circle(
        candidates,
        detection_rgb,
        edge if edge is not None else feature_edge(detection_rgb),
    )
    if texture_pad is not None:
        return texture_pad_mode, [texture_pad]

    return "fallback", []


def make_roi(shape, circles, scale, expansion=1.01):
    height, width = shape
    roi = np.zeros((height, width), dtype=np.uint8)
    for circle in circles:
        center = (int(round(circle["x"] / scale)), int(round(circle["y"] / scale)))
        radius = int(round(circle["radius"] * expansion / scale))
        cv2.circle(roi, center, radius, 1, thickness=cv2.FILLED)
    return roi


def apply_circle_prior_roi(
    rgb: np.ndarray,
    full_shape: Tuple[int, int],
    *,
    max_detection_dimension: int = 1200,
    expansion: float = 1.01,
    pad_aspect_min: float = 0.75,
    pad_aspect_max: float = 1.35,
    wide_aspect_min: float = 1.65,
) -> Tuple[Optional[np.ndarray], str]:
    """
    Detect circle ROI on a reduced thumbnail and map it to full_shape (height, width).

    Returns:
        (roi_mask, mode): roi_mask is uint8 with values 0/1, or None on fallback.
    """
    if rgb is None or rgb.size == 0:
        return None, "fallback"

    if len(rgb.shape) == 3 and rgb.shape[2] == 4:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)

    detection_rgb, scale = resize_for_detection(rgb, max_dimension=max_detection_dimension)
    edge = feature_edge(detection_rgb)
    raw_candidates = circle_candidates(detection_rgb, edge)
    candidates = [candidate_metrics(detection_rgb, edge, circle) for circle in raw_candidates]
    mode, accepted = select_circles_with_cascade(
        candidates,
        detection_rgb,
        edge=edge,
        pad_aspect_min=pad_aspect_min,
        pad_aspect_max=pad_aspect_max,
        wide_aspect_min=wide_aspect_min,
    )
    if not accepted:
        return None, mode

    roi = make_roi(full_shape, accepted, scale, expansion=expansion)
    return roi, mode


def density_mask_is_clearly_round(
    region_mask: np.ndarray,
    *,
    min_component_coverage: float = 0.85,
    min_aspect: float = 0.75,
    max_aspect: float = 1.35,
    min_circularity: float = 0.72,
    min_solidity: float = 0.90,
    min_fill_enclosing: float = 0.64,
    min_area_frac: float = 0.15,
    max_area_frac: float = 0.96,
) -> bool:
    """
    Return True when pixel-density segmentation already produced one obvious
    round-ish connected region, so pad/double-circle priors are unnecessary.
    """
    if region_mask is None or region_mask.size == 0:
        return False

    mask = (region_mask > 0).astype(np.uint8)
    total_pixels = int(mask.sum())
    if total_pixels == 0:
        return False

    area_frac = total_pixels / mask.size
    if not (min_area_frac <= area_frac <= max_area_frac):
        return False

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 100:
        return False

    if float(cv2.contourArea(contour)) / total_pixels < min_component_coverage:
        return False

    contour_area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    circularity = 4.0 * np.pi * contour_area / max(1e-6, perimeter * perimeter)
    if circularity < min_circularity:
        return False

    _, _, bbox_width, bbox_height = cv2.boundingRect(contour)
    aspect_ratio = bbox_width / max(1, bbox_height)
    if not (min_aspect <= aspect_ratio <= max_aspect):
        return False

    hull = cv2.convexHull(contour)
    solidity = contour_area / max(1e-6, float(cv2.contourArea(hull)))
    if solidity < min_solidity:
        return False

    _, radius = cv2.minEnclosingCircle(contour)
    fill_enclosing = contour_area / max(1e-6, np.pi * radius * radius)
    if fill_enclosing < min_fill_enclosing:
        return False

    return True


def density_mask_is_clearly_smear(
    region_mask: np.ndarray,
    *,
    min_component_coverage: float = 0.85,
    min_aspect: float = 1.35,
    max_aspect: float = 2.20,
    min_circularity: float = 0.55,
    min_solidity: float = 0.95,
    min_area_frac: float = 0.45,
    max_area_frac: float = 0.96,
) -> bool:
    """
    Return True for elongated single-band smear masks (thyroid/urinary style)
    where density segmentation is already reliable and circle prior would over-crop.
    """
    if region_mask is None or region_mask.size == 0:
        return False

    mask = (region_mask > 0).astype(np.uint8)
    total_pixels = int(mask.sum())
    if total_pixels == 0:
        return False

    area_frac = total_pixels / mask.size
    if not (min_area_frac <= area_frac <= max_area_frac):
        return False

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 100:
        return False

    if float(cv2.contourArea(contour)) / total_pixels < min_component_coverage:
        return False

    contour_area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    circularity = 4.0 * np.pi * contour_area / max(1e-6, perimeter * perimeter)
    if circularity < min_circularity:
        return False

    _, _, bbox_width, bbox_height = cv2.boundingRect(contour)
    aspect_ratio = bbox_width / max(1, bbox_height)
    if not (min_aspect <= aspect_ratio <= max_aspect):
        return False

    hull = cv2.convexHull(contour)
    solidity = contour_area / max(1e-6, float(cv2.contourArea(hull)))
    if solidity < min_solidity:
        return False

    return True


def single_roi_is_safe(rgb, region_mask, roi, min_retained_fraction=0.8, min_discarded_texture=14.0):
    selected = region_mask > 0
    selected_count = int(selected.sum())
    if selected_count == 0:
        return True
    retained_fraction = float(np.count_nonzero(selected & (roi > 0)) / selected_count)
    if retained_fraction >= min_retained_fraction:
        return True
    discarded = selected & (roi == 0)
    if not discarded.any():
        return True
    min_dim = min(rgb.shape[:2])
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sigma = max(1.5, min_dim * 0.006)
    mean = cv2.GaussianBlur(gray, (0, 0), sigma)
    mean_square = cv2.GaussianBlur(gray * gray, (0, 0), sigma)
    texture = np.sqrt(np.maximum(mean_square - mean * mean, 0))
    texture = cv2.GaussianBlur(texture, (0, 0), max(2.0, min_dim * 0.008))
    discarded_texture = float(np.median(texture[discarded]))
    return discarded_texture >= min_discarded_texture


def circle_roi_has_content_contrast(rgb, region_mask, roi, mode):
    scale = min(1.0, 1200.0 / max(rgb.shape[:2]))
    if scale < 1.0:
        size = (
            max(1, int(round(rgb.shape[1] * scale))),
            max(1, int(round(rgb.shape[0] * scale))),
        )
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        region_mask = cv2.resize(region_mask, size, interpolation=cv2.INTER_NEAREST)
        roi = cv2.resize(roi, size, interpolation=cv2.INTER_NEAREST)

    selected = region_mask > 0
    inside = selected & (roi > 0)
    outside = selected & (roi == 0)
    if not inside.any():
        return False
    if not outside.any():
        return True

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    chroma = np.hypot(lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0)
    stain_inside = float(np.mean(chroma[inside] >= 8.0))
    stain_outside = float(np.mean(chroma[outside] >= 8.0))
    if stain_inside - stain_outside >= 0.15:
        return True

    min_dim = min(rgb.shape[:2])
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sigma = max(1.5, min_dim * 0.006)
    mean = cv2.GaussianBlur(gray, (0, 0), sigma)
    mean_square = cv2.GaussianBlur(gray * gray, (0, 0), sigma)
    texture = np.sqrt(np.maximum(mean_square - mean * mean, 0))
    texture = cv2.GaussianBlur(texture, (0, 0), max(2.0, min_dim * 0.008))
    texture_inside = float(np.median(texture[inside]))
    texture_outside = float(np.median(texture[outside]))

    if mode.startswith("double"):
        return texture_inside - texture_outside >= 3.0 and texture_inside >= texture_outside * 1.5
    return texture_outside - texture_inside >= 3.0 and texture_outside >= texture_inside * 1.4


def _largest_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def enclosing_circle_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, float]]:
    contour = _largest_contour(mask)
    if contour is None or cv2.contourArea(contour) < 100:
        return None
    (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
    return int(round(center_x)), int(round(center_y)), float(radius)


def tissue_border_touch_count(mask: np.ndarray, margin: int) -> int:
    contour = _largest_contour(mask)
    if contour is None:
        return 0
    height, width = mask.shape[:2]
    xs = contour[:, 0, 0]
    ys = contour[:, 0, 1]
    touches = 0
    if int(xs.min()) <= margin:
        touches += 1
    if int(xs.max()) >= width - 1 - margin:
        touches += 1
    if int(ys.min()) <= margin:
        touches += 1
    if int(ys.max()) >= height - 1 - margin:
        touches += 1
    return touches


def _filled_circle_roi(shape, center_x, center_y, radius, expansion=1.0):
    height, width = shape
    roi = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(
        roi,
        (int(center_x), int(center_y)),
        int(round(radius * expansion)),
        1,
        thickness=cv2.FILLED,
    )
    return roi


def _mask_retained_fraction(base_mask: np.ndarray, roi: np.ndarray) -> float:
    base = base_mask > 0
    total = int(base.sum())
    if total == 0:
        return 1.0
    return float(np.count_nonzero(base & (roi > 0)) / total)


def refine_single_pad_with_enclosing(
    rgb: np.ndarray,
    base_mask: np.ndarray,
    final_mask: np.ndarray,
    circle_mode: str,
    *,
    expansion: float = 1.0,
    min_retained_gain: float = 0.03,
    min_area_gain: float = 0.03,
    min_border_touches: int = 2,
) -> Tuple[np.ndarray, str]:
    """
    Scheme B: for clipped/offset single pads, replace Hough ROI with
    minEnclosingCircle of the base tissue mask when Hough under-covers border tissue.
    """
    if circle_mode != "single":
        return final_mask, circle_mode

    enclosing = enclosing_circle_from_mask(base_mask)
    if enclosing is None:
        return final_mask, circle_mode

    center_x, center_y, radius = enclosing
    height, width = base_mask.shape[:2]
    margin = max(2, int(round(min(height, width) * 0.02)))

    enc_roi = _filled_circle_roi((height, width), center_x, center_y, radius, expansion)
    base_u8 = (base_mask > 0).astype(np.uint8)

    hough_retained = _mask_retained_fraction(base_mask, (final_mask > 0).astype(np.uint8))
    enc_retained = _mask_retained_fraction(base_mask, enc_roi)
    hough_area = float(np.mean(final_mask > 0))
    base_area = float(np.mean(base_mask > 0))
    enc_constrained = (base_u8 & enc_roi).astype(np.uint8) * 255
    enc_area = float(np.mean(enc_constrained > 0))

    border_touches = tissue_border_touch_count(base_mask, margin)
    retained_gain = enc_retained - hough_retained
    area_gain = enc_area - hough_area
    hough_to_base = hough_area / max(1e-6, base_area)

    use_enclosing = (
        border_touches >= min_border_touches
        and hough_to_base >= 0.84
        and hough_retained >= 0.82
        and retained_gain >= min_retained_gain
    ) or (
        area_gain >= min_area_gain and border_touches >= 1 and hough_to_base >= 0.88
    )

    if not use_enclosing or not single_roi_is_safe(rgb, base_mask, enc_roi):
        return final_mask, circle_mode

    return enc_constrained, "single_enclosing"
