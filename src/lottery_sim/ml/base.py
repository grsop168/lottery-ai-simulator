from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Protocol, Sequence, Tuple

from lottery_sim.models import Draw5D


PL5_POSITION_COUNT = 5
PL5_DIGIT_COUNT = 10
ProbabilityRow = Tuple[float, ...]
ProbabilityMatrix = Tuple[ProbabilityRow, ...]


def normalize_probability_row(values: Sequence[float]) -> ProbabilityRow:
    if len(values) != PL5_DIGIT_COUNT:
        raise ValueError("a PL5 probability row must contain digits 0 through 9")
    cleaned = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0 for value in cleaned):
        raise ValueError("probabilities must be finite and non-negative")
    total = sum(cleaned)
    if total <= 0:
        raise ValueError("probability row must have a positive sum")
    return tuple(value / total for value in cleaned)


@dataclass(frozen=True)
class ProbabilityPrediction:
    model_name: str
    probabilities: ProbabilityMatrix
    train_draw_count: int
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.probabilities) != PL5_POSITION_COUNT:
            raise ValueError("PL5 prediction must contain five positions")
        normalized = tuple(normalize_probability_row(row) for row in self.probabilities)
        object.__setattr__(self, "probabilities", normalized)
        if self.train_draw_count < 0:
            raise ValueError("train_draw_count cannot be negative")


class Pl5ProbabilityModel(Protocol):
    model_name: str

    def fit(self, history: Sequence[Draw5D]) -> "Pl5ProbabilityModel":
        ...

    def predict_proba(self, history: Sequence[Draw5D]) -> ProbabilityPrediction:
        ...
