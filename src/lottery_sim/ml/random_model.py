from __future__ import annotations

import hashlib
import random
from typing import List, Sequence

from lottery_sim.ml.base import ProbabilityPrediction
from lottery_sim.models import Draw5D


class RandomPl5Model:
    model_name = "random"

    def __init__(self, seed: int = 20260505):
        self.seed = int(seed)
        self.training_draw_count = 0

    def fit(self, history: Sequence[Draw5D]) -> "RandomPl5Model":
        self.training_draw_count = len(history)
        return self

    def predict_proba(self, history: Sequence[Draw5D]) -> ProbabilityPrediction:
        return ProbabilityPrediction(
            model_name=self.model_name,
            probabilities=tuple((0.1,) * 10 for _ in range(5)),
            train_draw_count=self.training_draw_count,
            metadata={"seed": self.seed, "probability_baseline": "uniform"},
        )

    def sample_candidates(self, target_issue: str, count: int) -> List[str]:
        if count < 1 or count > 100_000:
            raise ValueError("candidate count must be between 1 and 100000")
        digest = hashlib.sha256(f"{self.seed}:{target_issue}".encode("utf-8")).digest()
        issue_seed = int.from_bytes(digest[:8], "big")
        rng = random.Random(issue_seed)
        return [f"{value:05d}" for value in rng.sample(range(100_000), count)]
