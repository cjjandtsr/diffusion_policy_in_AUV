"""Typed boundary around the existing NumPy Reward v1.3 implementation."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class RewardContext:
    state: Dict[str, Any]
    action_chunk: np.ndarray
    next_state: Dict[str, Any]
    goal_world: np.ndarray
    obstacles: Any
    pointcloud: Optional[np.ndarray] = None
    offline_density: Optional[dict] = None
    config: Optional[dict] = None


@dataclass(frozen=True)
class RewardState:
    history: dict


@dataclass(frozen=True)
class RewardResult:
    total: float
    terms: Dict[str, float]
    metrics: Dict[str, Any]
    state: RewardState

