from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, Mapping

import numpy as np

from ..models import HeuristicDecision
from .base import decision, get_heuristic, prepare_thumbnail


def _attempt_record(result: HeuristicDecision) -> dict[str, Any]:
    return {
        "name": result.name,
        "status": result.status,
        "confidence": result.confidence,
        "reason": result.reason,
    }


def _validate_accepted(result: HeuristicDecision, shape: tuple[int, int]) -> str | None:
    if result.region is None:
        return "accepted decision has no region"
    mask = result.region.mask
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_:
        return "region mask must be a boolean NumPy array"
    if mask.ndim != 2 or mask.shape != shape:
        return f"region mask shape {getattr(mask, 'shape', None)} does not match {shape}"
    if result.region.thumbnail_size != (shape[1], shape[0]):
        return "RegionMask.thumbnail_size does not match the thumbnail"
    if not mask.any():
        return "region mask is empty"
    return None


class HeuristicPipeline:
    def __init__(
        self,
        names: Iterable[str],
        configs: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.names = tuple(names)
        if not self.names:
            raise ValueError("heuristic pipeline cannot be empty")
        self.configs = {name: dict(values) for name, values in (configs or {}).items()}

    @classmethod
    def from_config(cls, config: Any) -> "HeuristicPipeline":
        return cls(config.heuristic_pipe, config.heuristics)

    def run(
        self,
        thumbnail: np.ndarray,
        slide_size: tuple[int, int] | None = None,
    ) -> HeuristicDecision:
        try:
            image = prepare_thumbnail(thumbnail)
        except Exception as exc:
            return decision("pipeline", "error", f"Invalid thumbnail: {exc}")

        attempts: list[dict[str, Any]] = []
        for name in self.names:
            try:
                plugin = get_heuristic(name)
            except Exception as exc:
                result = decision(name, "error", str(exc))
                attempts.append(_attempt_record(result))
                continue

            result = plugin(image, slide_size, self.configs.get(name, {}))
            min_confidence = float(self.configs.get(name, {}).get("min_confidence", 0.0))
            if result.status == "accepted" and result.confidence < min_confidence:
                result = decision(
                    name,
                    "rejected",
                    f"Confidence {result.confidence:.3f} is below required {min_confidence:.3f}",
                    confidence=result.confidence,
                    metrics=result.metrics,
                )
            invalid = (
                _validate_accepted(result, image.shape[:2])
                if result.status == "accepted"
                else None
            )
            if result.status == "accepted" and invalid is None:
                metrics = dict(result.metrics)
                metrics["pipeline_attempts"] = attempts + [_attempt_record(result)]
                return replace(result, metrics=metrics)
            if invalid is not None:
                result = decision(
                    name,
                    "error",
                    f"Invalid accepted decision: {invalid}",
                    metrics=result.metrics,
                )
            attempts.append(_attempt_record(result))

        return decision(
            "pipeline",
            "rejected",
            "No configured heuristic produced a valid region",
            metrics={"pipeline_attempts": attempts},
        )


def run_pipeline(
    thumbnail: np.ndarray,
    slide_size: tuple[int, int] | None = None,
    heuristic_pipe: Iterable[str] | Any | None = None,
    heuristics: Mapping[str, Mapping[str, Any]] | None = None,
) -> HeuristicDecision:
    """Run names in order; ``heuristic_pipe`` may also be an AppConfig."""
    if heuristic_pipe is not None and hasattr(heuristic_pipe, "heuristic_pipe"):
        app_config = heuristic_pipe
        heuristic_pipe = app_config.heuristic_pipe
        heuristics = app_config.heuristics
    names = heuristic_pipe if heuristic_pipe is not None else ("density",)
    return HeuristicPipeline(names, heuristics).run(thumbnail, slide_size)


pipeline = run_pipeline
