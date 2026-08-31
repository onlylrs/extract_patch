from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ..models import PatchPlan


@dataclass(frozen=True)
class EncodedPatch:
    name: str
    data: bytes


def patch_stem(plan: PatchPlan) -> str:
    """Return a stable, coordinate-bearing patch identifier."""
    return f"{plan.x}_{plan.y}_{plan.output_size}"


def prepare_image(image: Image.Image, plan: PatchPlan) -> Image.Image:
    expected = (plan.output_size, plan.output_size)
    if plan.output_size <= 0:
        raise ValueError("Patch output_size must be positive")
    if image.size != expected:
        return image.resize(expected, Image.Resampling.LANCZOS)
    return image


class PatchSink(ABC):
    output_dir: Path

    @abstractmethod
    def encode(self, image: Image.Image, plan: PatchPlan) -> EncodedPatch:
        """Encode one image without publishing it."""

    @abstractmethod
    def publish(self, encoded: EncodedPatch) -> str:
        """Publish encoded bytes, returning the final output name."""

    def write(self, image: Image.Image, plan: PatchPlan) -> str:
        """Encode and publish one patch, returning its output name."""
        return self.publish(self.encode(image, plan))

    @abstractmethod
    def close(self) -> None:
        """Finish all pending output atomically."""

    def abort(self) -> None:
        self.close()

    def __enter__(self) -> "PatchSink":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()
