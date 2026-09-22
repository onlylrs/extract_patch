from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from extract_patch.filters.qc.model import preprocessing
from extract_patch.filters.qc.runtime import (
    QcSession,
    _images_to_uint8_batch,
    _normalize_uint8_batch,
)


def _random_images() -> list[Image.Image]:
    generator = np.random.default_rng(20260922)
    return [
        Image.fromarray(
            generator.integers(0, 256, size=(height, width, 3), dtype=np.uint8),
            mode="RGB",
        )
        for width, height in ((300, 300), (301, 299), (1200, 900), (256, 256))
    ]


def test_uint8_transport_is_four_times_smaller_and_bitwise_equivalent() -> None:
    images = _random_images()
    legacy_transform = preprocessing(256)
    legacy = torch.stack([legacy_transform(image) for image in images])

    compact = _images_to_uint8_batch(images, 256)
    restored = _normalize_uint8_batch(compact)

    assert compact.dtype == np.uint8
    assert legacy.numpy().nbytes == compact.nbytes * 4
    assert torch.equal(restored, legacy)


def test_uint8_and_legacy_paths_produce_identical_scores_and_decisions() -> None:
    images = _random_images()
    legacy = torch.stack([preprocessing(256)(image) for image in images])
    compact = _images_to_uint8_batch(images, 256)

    class DeterministicModel(torch.nn.Module):
        def forward(self, tensor):
            value = tensor.mean(dim=(1, 2, 3))
            return torch.stack((-value, value), dim=1)

    session = object.__new__(QcSession)
    session._torch = torch
    session.device = torch.device("cpu")
    session.model = DeterministicModel().eval()

    legacy_scores = session.predict_numpy(legacy.numpy())
    compact_scores = session.predict_uint8(compact)
    threshold = 0.5

    assert compact_scores == legacy_scores
    assert [score >= threshold for score in compact_scores] == [
        score >= threshold for score in legacy_scores
    ]
