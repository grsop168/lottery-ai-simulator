from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Sequence, Tuple

from lottery_sim.backtest.pl5_metrics import EvaluatedPrediction, Pl5ModelMetrics, calculate_pl5_metrics
from lottery_sim.candidate.probability_generator import generate_probability_candidates
from lottery_sim.ml.base import Pl5ProbabilityModel
from lottery_sim.ml.random_model import RandomPl5Model
from lottery_sim.models import Draw5D


ModelFactory = Callable[[], Pl5ProbabilityModel]


@dataclass(frozen=True)
class Pl5WalkForwardResult:
    metrics: Pl5ModelMetrics
    target_issues: Tuple[str, ...]
    training_cutoffs: Tuple[str, ...]
    retrained: Tuple[bool, ...]


def run_pl5_walk_forward(
    draws: Sequence[Draw5D],
    model_factory: ModelFactory,
    min_train_draws: int = 300,
    backtest_draws: int = 500,
    retrain_every: int = 20,
    candidate_count: int = 100,
    top_k_values: Sequence[int] = (10, 50, 100),
) -> Pl5WalkForwardResult:
    if min_train_draws < 1 or retrain_every < 1:
        raise ValueError("min_train_draws and retrain_every must be positive")
    required_candidates = max((candidate_count, *top_k_values))
    ordered = tuple(sorted(draws, key=lambda draw: int(draw.issue)))
    if len(ordered) <= min_train_draws:
        raise ValueError("not enough draws for walk-forward backtest")
    start = max(min_train_draws, len(ordered) - backtest_draws if backtest_draws > 0 else min_train_draws)
    evaluated: List[EvaluatedPrediction] = []
    target_issues: List[str] = []
    training_cutoffs: List[str] = []
    retrained_flags: List[bool] = []
    model: Pl5ProbabilityModel | None = None
    last_trained_index = -1

    for target_index in range(start, len(ordered)):
        history = ordered[:target_index]
        retrained = model is None or target_index - last_trained_index >= retrain_every
        if retrained:
            model = model_factory()
            model.fit(history)
            last_trained_index = target_index
        prediction = model.predict_proba(history)
        target = ordered[target_index]
        if isinstance(model, RandomPl5Model):
            candidates = tuple(model.sample_candidates(target.issue, required_candidates))
        else:
            candidates = tuple(candidate.number_text for candidate in generate_probability_candidates(prediction, required_candidates))
        evaluated.append(EvaluatedPrediction(prediction, target.numbers, candidates))
        target_issues.append(target.issue)
        training_cutoffs.append(ordered[last_trained_index - 1].issue)
        retrained_flags.append(retrained)

    model_name = evaluated[0].prediction.model_name if evaluated else "unknown"
    return Pl5WalkForwardResult(
        metrics=calculate_pl5_metrics(model_name, evaluated, top_k_values),
        target_issues=tuple(target_issues),
        training_cutoffs=tuple(training_cutoffs),
        retrained=tuple(retrained_flags),
    )
