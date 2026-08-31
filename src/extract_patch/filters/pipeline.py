from __future__ import annotations

from collections.abc import Iterable, Mapping
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
        for name in self.names:
            patch_filter = get_filter(name)
            result = patch_filter(image, plan, self.configs.get(name, {}))
            if not result.keep:
                return result
        return FilterDecision(
            keep=True,
            name="pipeline",
            reason="All configured post-extraction filters accepted the patch",
            metrics={"filters": list(self.names)},
        )
