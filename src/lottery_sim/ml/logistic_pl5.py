from __future__ import annotations

from typing import Sequence

from lottery_sim.ml.base import ProbabilityPrediction
from lottery_sim.ml.generic import (
    GenericMlModel,
    _score_group,
    ml_adapter_for_game,
    train_generic_ml_model,
)
from lottery_sim.models import Draw5D


class LogisticPl5Model:
    model_name = "logistic"

    def __init__(self, min_history: int = 30, epochs: int = 30, learning_rate: float = 0.04):
        self.min_history = int(min_history)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.model: GenericMlModel | None = None
        self.adapter = ml_adapter_for_game("pl5")

    def fit(self, history: Sequence[Draw5D]) -> "LogisticPl5Model":
        self.model = train_generic_ml_model(
            history,
            self.adapter,
            min_history=self.min_history,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
        )
        return self

    def predict_proba(self, history: Sequence[Draw5D]) -> ProbabilityPrediction:
        if self.model is None:
            raise ValueError("logistic model must be fitted before prediction")
        ordered = tuple(sorted(history, key=lambda draw: int(draw.issue)))
        rows = []
        for index, group_model in enumerate(self.model.groups):
            scores = _score_group(ordered, self.adapter, index, group_model, self.model.windows)
            score_map = dict(scores)
            rows.append(tuple(score_map[digit] for digit in range(10)))
        return ProbabilityPrediction(
            self.model_name,
            tuple(rows),
            self.model.training_draw_count,
            metadata={"min_history": self.min_history, "epochs": self.epochs},
        )
