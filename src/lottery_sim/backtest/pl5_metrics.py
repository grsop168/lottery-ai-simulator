from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from lottery_sim.ml.base import ProbabilityPrediction


@dataclass(frozen=True)
class ReturnMetrics:
    candidate_count: int
    total_cost: float
    total_payout: float
    profit: float
    return_rate: float
    profit_roi: float


@dataclass(frozen=True)
class Pl5ModelMetrics:
    model_name: str
    backtest_draws: int
    position_top1_accuracy: Tuple[float, ...]
    mean_position_accuracy: float
    position_top3_coverage: Tuple[float, ...]
    mean_top3_coverage: float
    position_top5_coverage: Tuple[float, ...]
    mean_top5_coverage: float
    average_correct_positions: float
    position_hit_distribution: Dict[str, int]
    exact_hits: int
    exact_hit_rate: float
    log_loss: float
    brier_score: float
    candidate_exact_coverage: Dict[str, float]
    returns: Dict[str, ReturnMetrics]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluatedPrediction:
    prediction: ProbabilityPrediction
    actual: Tuple[int, int, int, int, int]
    candidate_numbers: Tuple[str, ...]


def calculate_pl5_metrics(
    model_name: str,
    evaluated: Sequence[EvaluatedPrediction],
    top_k_values: Sequence[int] = (10, 50, 100),
    ticket_cost: float = 2.0,
    direct_prize: float = 100_000.0,
    epsilon: float = 1e-15,
) -> Pl5ModelMetrics:
    draw_count = len(evaluated)
    top1_hits = [0] * 5
    top3_hits = [0] * 5
    top5_hits = [0] * 5
    distribution = {str(value): 0 for value in range(6)}
    log_loss_total = 0.0
    brier_total = 0.0
    exact_hits = 0
    candidate_hits = {int(value): 0 for value in top_k_values}

    for item in evaluated:
        correct_positions = 0
        for position, actual_digit in enumerate(item.actual):
            row = item.prediction.probabilities[position]
            ranking = sorted(range(10), key=lambda digit: (-row[digit], digit))
            if ranking[0] == actual_digit:
                top1_hits[position] += 1
                correct_positions += 1
            if actual_digit in ranking[:3]:
                top3_hits[position] += 1
            if actual_digit in ranking[:5]:
                top5_hits[position] += 1
            log_loss_total -= math.log(max(row[actual_digit], epsilon))
            brier_total += sum((row[digit] - (1.0 if digit == actual_digit else 0.0)) ** 2 for digit in range(10))
        distribution[str(correct_positions)] += 1
        if correct_positions == 5:
            exact_hits += 1
        actual_text = "".join(str(value) for value in item.actual)
        for top_k in candidate_hits:
            if actual_text in item.candidate_numbers[:top_k]:
                candidate_hits[top_k] += 1

    denominator = draw_count or 1
    position_denominator = denominator
    position_top1 = tuple(value / position_denominator for value in top1_hits)
    position_top3 = tuple(value / position_denominator for value in top3_hits)
    position_top5 = tuple(value / position_denominator for value in top5_hits)
    returns = {}
    for top_k, hits in candidate_hits.items():
        total_cost = draw_count * top_k * ticket_cost
        total_payout = hits * direct_prize
        profit = total_payout - total_cost
        returns[f"top{top_k}"] = ReturnMetrics(
            candidate_count=top_k,
            total_cost=total_cost,
            total_payout=total_payout,
            profit=profit,
            return_rate=total_payout / total_cost if total_cost else 0.0,
            profit_roi=profit / total_cost if total_cost else 0.0,
        )
    return Pl5ModelMetrics(
        model_name=model_name,
        backtest_draws=draw_count,
        position_top1_accuracy=position_top1,
        mean_position_accuracy=sum(position_top1) / 5,
        position_top3_coverage=position_top3,
        mean_top3_coverage=sum(position_top3) / 5,
        position_top5_coverage=position_top5,
        mean_top5_coverage=sum(position_top5) / 5,
        average_correct_positions=sum(int(key) * count for key, count in distribution.items()) / denominator,
        position_hit_distribution=distribution,
        exact_hits=exact_hits,
        exact_hit_rate=exact_hits / denominator,
        log_loss=log_loss_total / (denominator * 5),
        brier_score=brier_total / (denominator * 5),
        candidate_exact_coverage={f"top{key}": value / denominator for key, value in candidate_hits.items()},
        returns=returns,
    )
