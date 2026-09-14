from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from PIL import Image

from ..models import PatchPlan
from .base import FilterDecision, get_filter


class PatchFilterPipeline:
    """Apply post-extraction filters in order; the first rejection drops a patch."""

    def __init__(
        self,
        names: Iterable[str] = (),
        configs: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.names = tuple(names)
        self.configs = {name: dict(values) for name, values in (configs or {}).items()}

    @classmethod
    def from_config(cls, config: Any) -> "PatchFilterPipeline":
        return cls(config.post_filter_pipe, config.post_filters)

    def run(self, image: Image.Image, plan: PatchPlan) -> FilterDecision:
        decisions = self.run_many(((image, plan),))
        return decisions[0]

    def run_many(
        self,
        items: Sequence[tuple[Image.Image, PatchPlan]],
    ) -> list[FilterDecision]:
        if not items:
            return []
        accepted = FilterDecision(
            keep=True,
            name="pipeline",
            reason="All configured post-extraction filters accepted the patch",
            metrics={"filters": list(self.names)},
        )
        if not self.names:
            return [accepted] * len(items)

        decisions: list[FilterDecision | None] = [None] * len(items)
        active = list(range(len(items)))
        for name in self.names:
            if not active:
                break
            patch_filter = get_filter(name)
            results = patch_filter.evaluate_many(
                [items[index] for index in active],
                self.configs.get(name, {}),
            )
            if len(results) != len(active):
                raise RuntimeError(
                    f"{name} returned {len(results)} decisions for {len(active)} patches"
                )
            next_active: list[int] = []
            for index, result in zip(active, results):
                if result.keep:
                    next_active.append(index)
                else:
                    decisions[index] = result
            active = next_active
        for index in active:
            decisions[index] = accepted
        return [decision if decision is not None else accepted for decision in decisions]
