"""Configurable thumbnail heuristics for selecting WSI extraction regions."""

from .base import Heuristic, get_heuristic, register, registry
from .circle import (
    DoubleCircleHeuristic,
    SingleCircleHeuristic,
    SmearHeuristic,
    double_circle,
    single_circle,
    smear,
)
from .density import DensityHeuristic, density
from .pipeline import HeuristicPipeline, pipeline, run_pipeline
from .qmh import QmhCenterCircleHeuristic, qmh_center_circle
from .serrated import SerratedOuterCircleHeuristic, serrated_outer_circle
from .smartcyto import SmartCytoCircleHeuristic, smartcyto_circle

__all__ = [
    "DensityHeuristic",
    "DoubleCircleHeuristic",
    "Heuristic",
    "HeuristicPipeline",
    "QmhCenterCircleHeuristic",
    "SerratedOuterCircleHeuristic",
    "SingleCircleHeuristic",
    "SmartCytoCircleHeuristic",
    "SmearHeuristic",
    "density",
    "double_circle",
    "get_heuristic",
    "pipeline",
    "qmh_center_circle",
    "register",
    "registry",
    "run_pipeline",
    "serrated_outer_circle",
    "single_circle",
    "smartcyto_circle",
    "smear",
]
