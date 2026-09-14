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
        "metrics": dict(result.metrics),
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
        strategy: str = "first",
        arbitration: Mapping[str, Any] | None = None,
    ) -> None:
        self.names = tuple(names)
        if not self.names:
            raise ValueError("heuristic pipeline cannot be empty")
        if strategy not in {"first", "arbitrate"}:
            raise ValueError("heuristic strategy must be first or arbitrate")
        self.configs = {name: dict(values) for name, values in (configs or {}).items()}
        self.strategy = strategy
        self.arbitration = dict(arbitration or {})

    @classmethod
    def from_config(cls, config: Any) -> "HeuristicPipeline":
        return cls(
            config.heuristic_pipe,
            config.heuristics,
            getattr(config, "heuristic_strategy", "first"),
            getattr(config, "heuristic_arbitration", {}),
        )

    def _arbitrate(
        self,
        accepted: list[HeuristicDecision],
    ) -> tuple[HeuristicDecision, dict[str, Any]]:
        density = next((result for result in accepted if result.name == "density"), None)
        if density is None or density.region is None:
            return accepted[0], {
                "selected": accepted[0].name,
                "reason": "No valid density reference; selected first valid specialized proposal",
                "candidates": [],
            }

        reference = density.region.mask
        reference_area = int(np.count_nonzero(reference))
        candidates: list[dict[str, Any]] = []
        safe_results: list[HeuristicDecision] = []
        for result in accepted:
            if result is density or result.region is None:
                continue
            mask = result.region.mask
            candidate_area = int(np.count_nonzero(mask))
            overlap = int(np.count_nonzero(mask & reference))
            density_recall = overlap / max(1, reference_area)
            area_ratio = candidate_area / max(1, reference_area)
            cfg = self.configs.get(result.name, {})
            min_confidence = float(
                cfg.get(
                    "arbitration_min_confidence",
                    self.arbitration.get("min_specialized_confidence", 0.65),
                )
            )
            min_density_recall = float(
                cfg.get(
                    "min_density_recall",
                    self.arbitration.get("min_density_recall", 0.90),
                )
            )
            max_area_ratio = float(
                cfg.get(
                    "max_density_area_ratio",
                    self.arbitration.get("max_density_area_ratio", 3.5),
                )
            )
            density_guard = bool(cfg.get("density_guard", True))
            safe = (
                result.confidence >= min_confidence
                and (
                    not density_guard
                    or (
                        density_recall >= min_density_recall
                        and area_ratio <= max_area_ratio
                    )
                )
            )
            record = {
                "name": result.name,
                "confidence": result.confidence,
                "density_guard": density_guard,
                "density_recall": density_recall,
                "density_area_ratio": area_ratio,
                "required_confidence": min_confidence,
                "required_density_recall": min_density_recall,
                "maximum_density_area_ratio": max_area_ratio,
                "safe": safe,
            }
            candidates.append(record)
            if safe:
                safe_results.append(result)
        if safe_results:
            selected = max(safe_results, key=lambda result: result.confidence)
            return selected, {
                "selected": selected.name,
                "reason": (
                    "Selected highest-confidence specialized proposal among "
                    "candidates that passed safety gates"
                ),
                "candidates": candidates,
            }
        return density, {
            "selected": density.name,
            "reason": "Specialized proposals failed density safety gates",
            "candidates": candidates,
        }

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
        accepted: list[HeuristicDecision] = []
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
                attempts.append(_attempt_record(result))
                if self.strategy == "first":
                    metrics = dict(result.metrics)
                    metrics["pipeline_strategy"] = self.strategy
                    metrics["pipeline_attempts"] = attempts
                    return replace(result, metrics=metrics)
                accepted.append(result)
                continue
            if invalid is not None:
                result = decision(
                    name,
                    "error",
                    f"Invalid accepted decision: {invalid}",
                    metrics=result.metrics,
                )
            attempts.append(_attempt_record(result))

        if accepted:
            selected, arbitration = self._arbitrate(accepted)
            metrics = dict(selected.metrics)
            metrics["pipeline_strategy"] = self.strategy
            metrics["pipeline_attempts"] = attempts
            metrics["pipeline_arbitration"] = arbitration
            return replace(selected, metrics=metrics)
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
    strategy: str = "first",
    arbitration: Mapping[str, Any] | None = None,
) -> HeuristicDecision:
    """Run names in order; ``heuristic_pipe`` may also be an AppConfig."""
    if heuristic_pipe is not None and hasattr(heuristic_pipe, "heuristic_pipe"):
        app_config = heuristic_pipe
        heuristic_pipe = app_config.heuristic_pipe
        heuristics = app_config.heuristics
        strategy = getattr(app_config, "heuristic_strategy", "first")
        arbitration = getattr(app_config, "heuristic_arbitration", {})
    names = heuristic_pipe if heuristic_pipe is not None else ("density",)
    return HeuristicPipeline(names, heuristics, strategy, arbitration).run(
        thumbnail, slide_size
    )


pipeline = run_pipeline
