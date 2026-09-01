from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

from lottery_sim.ml.base import ProbabilityPrediction
from lottery_sim.ml.pl5_features import FEATURE_WINDOWS, build_pl5_prefix_features, build_pl5_training_rows
from lottery_sim.models import Draw5D


class LightGbmPl5Model:
    model_name = "lightgbm"

    def __init__(
        self,
        min_history: int = 30,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = -1,
        random_state: int = 20260505,
    ):
        self.min_history = int(min_history)
        self.parameters: Dict[str, Any] = {
            "objective": "multiclass",
            "num_class": 10,
            "n_estimators": int(n_estimators),
            "learning_rate": float(learning_rate),
            "num_leaves": int(num_leaves),
            "max_depth": int(max_depth),
            "random_state": int(random_state),
            "verbosity": -1,
            "n_jobs": 1,
        }
        self.models: List[Any] = []
        self.training_draw_count = 0

    def fit(self, history: Sequence[Draw5D]) -> "LightGbmPl5Model":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise RuntimeError("lightgbm is required for the LightGBM PL5 model") from exc
        ordered = tuple(sorted(history, key=lambda draw: int(draw.issue)))
        if len(ordered) <= self.min_history:
            raise ValueError("not enough history to train LightGBM")
        self.models = []
        for position in range(5):
            rows = build_pl5_training_rows(ordered, position, self.min_history, FEATURE_WINDOWS)
            classifier = LGBMClassifier(**self.parameters)
            classifier.fit(rows.features, rows.targets)
            self.models.append(classifier)
        self.training_draw_count = len(ordered)
        return self

    def predict_proba(self, history: Sequence[Draw5D]) -> ProbabilityPrediction:
        if len(self.models) != 5:
            raise ValueError("LightGBM model must be fitted before prediction")
        rows = []
        for position, classifier in enumerate(self.models):
            features = build_pl5_prefix_features(history, position, FEATURE_WINDOWS)
            raw = classifier.predict_proba([features])[0]
            mapped = [0.0] * 10
            for class_value, probability in zip(classifier.classes_, raw):
                mapped[int(class_value)] = float(probability)
            rows.append(tuple(mapped))
        return ProbabilityPrediction(
            self.model_name,
            tuple(rows),
            self.training_draw_count,
            metadata={"parameters": dict(self.parameters), "min_history": self.min_history},
        )

    def save(self, directory: Path) -> None:
        if len(self.models) != 5:
            raise ValueError("LightGBM model must be fitted before saving")
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        for position, classifier in enumerate(self.models):
            classifier.booster_.save_model(str(path / f"position-{position + 1}.txt"))
