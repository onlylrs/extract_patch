from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, TypeVar

from PIL import Image

from ..models import PatchPlan

Config = Mapping[str, Any]
F = TypeVar("F", bound="PatchFilter")

_REGISTRY: dict[str, type["PatchFilter"]] = {}


@dataclass(frozen=True)
class FilterDecision:
    keep: bool
    reason: str
    metrics: dict[str, Any] = field(default_factory=dict)
    name: str = ""


class PatchFilter(ABC):
    name = ""

    @abstractmethod
    def evaluate(
        self,
        image: Image.Image,
        plan: PatchPlan,
        config: Config | None = None,
    ) -> FilterDecision:
        """Return whether one decoded patch should be kept."""

    def __call__(
        self,
        image: Image.Image,
        plan: PatchPlan,
        config: Config | None = None,
    ) -> FilterDecision:
        try:
            return self.evaluate(image, plan, config)
        except Exception as exc:
            return FilterDecision(
                keep=False,
                name=self.name,
                reason=f"{type(exc).__name__}: {exc}",
                metrics={"exception_type": type(exc).__name__},
            )


def register(cls: type[F]) -> type[F]:
    if not cls.name:
        raise ValueError("A patch filter must define a non-empty name")
    if cls.name in _REGISTRY and _REGISTRY[cls.name] is not cls:
        raise ValueError(f"Patch filter already registered: {cls.name}")
    _REGISTRY[cls.name] = cls
    return cls


def registry() -> dict[str, type["PatchFilter"]]:
    return dict(_REGISTRY)


def get_filter(name: str) -> PatchFilter:
    try:
        return _REGISTRY[name]()
    except KeyError as exc:
        raise KeyError(
            f"Unknown patch filter {name!r}; available: {sorted(_REGISTRY)}"
        ) from exc
