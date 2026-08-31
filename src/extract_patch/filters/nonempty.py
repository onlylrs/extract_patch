from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from ..heuristics.base import prepare_thumbnail
from ..models import PatchPlan
from .base import Config, FilterDecision, PatchFilter, register


@register
class NonEmptyPatchFilter(PatchFilter):
    """Drop only patches with no defensible cellular or texture evidence."""

    name = "nonempty"

    def evaluate(
        self,
        image: Image.Image,
        plan: PatchPlan,
        config: Config | None = None,
    ) -> FilterDecision:
        cfg = config or {}
        rgb = prepare_thumbnail(np.asarray(image.convert("RGB")))
        max_dimension = int(cfg.get("max_analysis_dimension", 320))
        scale = min(1.0, max_dimension / max(rgb.shape[:2]))
        analysis = (
            cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if scale < 1.0
            else rgb
        )
        min_dim = min(analysis.shape[:2])

        # Subtract a smooth local color field. Constant white/gray/dark
        # backgrounds and illumination gradients disappear, while cells and
        # staining defects remain as localized Lab-color protrusions.
        lab = cv2.cvtColor(analysis, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab = cv2.GaussianBlur(
            lab,
            (0, 0),
            float(cfg.get("noise_sigma", 0.8)),
        )
        background = cv2.GaussianBlur(
            lab,
            (0, 0),
            max(
                3.0,
                min_dim * float(cfg.get("background_sigma_ratio", 0.035)),
            ),
        )
        residual = lab - background
        local_color_delta = np.sqrt(
            0.25 * residual[:, :, 0] ** 2
            + residual[:, :, 1] ** 2
            + residual[:, :, 2] ** 2
        )
        color_threshold = float(cfg.get("local_color_delta", 3.0))
        local_color_fraction = float(np.mean(local_color_delta >= color_threshold))

        # A Laplacian texture score rejects smooth gradients even when their
        # global standard deviation is large.
        gray = cv2.cvtColor(analysis, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray = cv2.GaussianBlur(
            gray,
            (0, 0),
            float(cfg.get("noise_sigma", 0.8)),
        )
        laplacian_std = float(cv2.Laplacian(gray, cv2.CV_32F, ksize=3).std())
        near_white_fraction = float(np.mean(np.all(rgb >= 245, axis=2)))

        keep = (
            local_color_fraction
            >= float(cfg.get("min_local_color_fraction", 0.001))
            or laplacian_std >= float(cfg.get("min_laplacian_std", 1.9))
        )
        metrics = {
            "analysis_scale": scale,
            "local_color_fraction": local_color_fraction,
            "local_color_p99": float(np.percentile(local_color_delta, 99)),
            "laplacian_std": laplacian_std,
            "near_white_fraction": near_white_fraction,
        }
        if keep:
            return FilterDecision(
                keep=True,
                name=self.name,
                reason="Patch contains localized color or texture evidence",
                metrics=metrics,
            )
        return FilterDecision(
            keep=False,
            name=self.name,
            reason="Patch is smooth pure-color or gradient background",
            metrics=metrics,
        )


def nonempty(
    image: Image.Image,
    plan: PatchPlan,
    config: Config | None = None,
) -> FilterDecision:
    return NonEmptyPatchFilter()(image, plan, config)
