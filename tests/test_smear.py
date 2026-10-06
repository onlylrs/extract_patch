import cv2
import numpy as np
import pytest

from extract_patch.heuristics import smear


@pytest.mark.parametrize("shape", [(240, 240), (240, 300), (300, 240), (240, 480)])
def test_full_field_stain_does_not_require_elongated_canvas(shape):
    image = np.full((*shape, 3), (190, 70, 160), dtype=np.uint8)
    image[np.random.default_rng(5).random(shape) < 0.25] = 245
    result = smear(image)
    assert result.status == "accepted"
    assert result.metrics["mode"] == "full_field"
    assert result.region.mask.mean() >= 0.99
    assert smear(image, config={"allow_full_field": False}).status == "not_applicable"


def test_full_field_texture_with_circular_stain_is_not_smear():
    gray = np.random.default_rng(11).integers(130, 256, (240, 300), dtype=np.uint8)
    image = np.repeat(gray[:, :, None], 3, axis=2)
    cv2.circle(image, (150, 120), 85, (190, 70, 160), cv2.FILLED)
    result = smear(image)
    assert result.metrics["area_fraction"] > 0.90
    assert result.status == "not_applicable"


def test_standard_elongated_smear_and_blank_keep_their_behavior():
    image = np.full((240, 480, 3), 255, dtype=np.uint8)
    assert smear(image).status == "not_applicable"
    cv2.ellipse(image, (240, 120), (200, 65), 0, 0, 360, (190, 70, 160), -1)
    result = smear(image)
    assert result.status == "accepted"
    assert result.metrics["mode"] == "standard"
