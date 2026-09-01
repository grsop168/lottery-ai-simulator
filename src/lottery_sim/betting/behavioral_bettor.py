from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Dict, Sequence, Tuple

import numpy as np

from lottery_sim.models import Draw5D


@dataclass(frozen=True)
class BehavioralBettorConfig:
    preferred_digit_weight: float = 0.18
    avoided_four_weight: float = -0.30
    repeated_digit_weight: float = 0.18
    adjacent_sequence_weight: float = 0.12
    full_straight_weight: float = 0.45
    symmetry_weight: float = 0.35
    patterned_pick_weight: float = 0.35
    recent_draw_match_weight: float = 0.16
    hot_digit_weight: float = 0.22
    cold_digit_weight: float = -0.08
    fixed_number_weight: float = 0.12
    history_window: int = 60
    random_seed: int = 20260505


class BehavioralBettor:
    """Parameterized synthetic bettor behavior; it is not an estimate of real bettor shares."""

    name = "behavioral"

    def __init__(self, config: BehavioralBettorConfig | None = None):
        self.config = config or BehavioralBettorConfig()

    @property
    def parameters(self) -> Dict[str, object]:
        return asdict(self.config)

    def manual_probabilities(self, history: Sequence[Draw5D]) -> np.ndarray:
        digits, static_score = _number_space(self.config)
        score = static_score.copy()
        # Walk-forward callers provide chronological prefixes. Avoid sorting/scanning the
        # entire prefix here because this estimator is evaluated once per historical draw.
        ordered = history
        if ordered:
            latest = np.asarray(ordered[-1].numbers, dtype=np.int8)
            score += self.config.recent_draw_match_weight * (digits == latest).sum(axis=1)
            scoped = ordered[-max(1, self.config.history_window):]
            for position in range(5):
                counts = np.bincount([draw.numbers[position] for draw in scoped], minlength=10).astype(np.float64)
                frequencies = counts / len(scoped)
                centered = frequencies - 0.1
                positive = np.maximum(centered, 0.0)
                negative = np.maximum(-centered, 0.0)
                score += self.config.hot_digit_weight * positive[digits[:, position]]
                score += self.config.cold_digit_weight * negative[digits[:, position]]
        score -= float(score.max())
        weights = np.exp(score)
        return weights / weights.sum()


@lru_cache(maxsize=8)
def _number_space(config: BehavioralBettorConfig) -> Tuple[np.ndarray, np.ndarray]:
    numbers = np.arange(100_000, dtype=np.int32)
    divisors = np.asarray((10_000, 1_000, 100, 10, 1), dtype=np.int32)
    digits = ((numbers[:, None] // divisors) % 10).astype(np.int8)
    score = np.zeros(100_000, dtype=np.float64)
    score += config.preferred_digit_weight * np.isin(digits, (6, 8, 9)).sum(axis=1)
    score += config.avoided_four_weight * (digits == 4).sum(axis=1)
    unique_counts = np.apply_along_axis(lambda row: len(set(row.tolist())), 1, digits)
    score += config.repeated_digit_weight * (5 - unique_counts)
    adjacent = (np.abs(np.diff(digits.astype(np.int16), axis=1)) == 1).sum(axis=1)
    score += config.adjacent_sequence_weight * adjacent
    diffs = np.diff(digits.astype(np.int16), axis=1)
    score += config.full_straight_weight * (np.all(diffs == 1, axis=1) | np.all(diffs == -1, axis=1))
    score += config.symmetry_weight * np.all(digits == digits[:, ::-1], axis=1)
    ababa = (digits[:, 0] == digits[:, 2]) & (digits[:, 2] == digits[:, 4]) & (digits[:, 1] == digits[:, 3])
    aabba = (digits[:, 0] == digits[:, 1]) & (digits[:, 3] == digits[:, 4])
    score += config.patterned_pick_weight * (ababa | aabba)
    rng = np.random.default_rng(config.random_seed)
    score += config.fixed_number_weight * rng.standard_normal(100_000)
    digits.setflags(write=False)
    score.setflags(write=False)
    return digits, score
