from __future__ import annotations

from collections import Counter
from typing import Sequence, Tuple

from lottery_sim.ml.base import ProbabilityPrediction
from lottery_sim.models import Draw5D


class MarkovPl5Model:
    model_name = "markov"

    def __init__(self, smoothing: float = 1.0, min_history: int = 30):
        if smoothing <= 0:
            raise ValueError("smoothing must be positive")
        self.smoothing = float(smoothing)
        self.min_history = int(min_history)
        self.training_draw_count = 0
        self.transitions: Tuple[Tuple[Tuple[float, ...], ...], ...] = ()
        self.frequencies: Tuple[Tuple[float, ...], ...] = ()

    def fit(self, history: Sequence[Draw5D]) -> "MarkovPl5Model":
        ordered = tuple(sorted(history, key=lambda draw: int(draw.issue)))
        self.training_draw_count = len(ordered)
        position_transitions = []
        position_frequencies = []
        for position in range(5):
            counts = [[self.smoothing for _ in range(10)] for _ in range(10)]
            digits = [draw.numbers[position] for draw in ordered]
            for previous, following in zip(digits, digits[1:]):
                counts[previous][following] += 1.0
            position_transitions.append(tuple(tuple(value / sum(row) for value in row) for row in counts))
            frequency_counts = Counter(digits)
            denominator = len(digits) + self.smoothing * 10
            position_frequencies.append(tuple((frequency_counts[digit] + self.smoothing) / denominator for digit in range(10)))
        self.transitions = tuple(position_transitions)
        self.frequencies = tuple(position_frequencies)
        return self

    def predict_proba(self, history: Sequence[Draw5D]) -> ProbabilityPrediction:
        if not self.transitions:
            raise ValueError("markov model must be fitted before prediction")
        ordered = tuple(sorted(history, key=lambda draw: int(draw.issue)))
        use_fallback = len(ordered) < self.min_history or not ordered
        if use_fallback:
            probabilities = self.frequencies
        else:
            probabilities = tuple(
                self.transitions[position][ordered[-1].numbers[position]]
                for position in range(5)
            )
        return ProbabilityPrediction(
            self.model_name,
            probabilities,
            self.training_draw_count,
            metadata={"order": 1, "smoothing": self.smoothing, "frequency_fallback": use_fallback},
        )
