"""
数据结构

Lightweight dataclasses shared across the pattern clustering pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass(slots=True)
class Thesis:
    """Single thesis (argument) extracted from LLM analysis."""

    stock: str
    date: int
    analyst: str
    polarity: bool
    content: str
    embedding: Optional[np.ndarray] = None
    score: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ThesisPatternRecord:
    """Stores the mapping between a thesis and the current pattern mixture."""

    stock: str
    date: int
    polarity_sign: float
    responsibilities: np.ndarray
