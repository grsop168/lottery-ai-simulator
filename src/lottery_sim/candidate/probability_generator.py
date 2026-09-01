from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import List, Tuple

from lottery_sim.ml.base import ProbabilityPrediction


@dataclass(frozen=True)
class ProbabilityCandidate:
    rank: int
    number_text: str
    model_name: str
    probability: float
    log_probability: float
    position_probabilities: Tuple[float, ...]
    reason: str


def generate_probability_candidates(prediction: ProbabilityPrediction, count: int = 10) -> List[ProbabilityCandidate]:
    if count < 1 or count > 100_000:
        raise ValueError("candidate count must be between 1 and 100000")
    epsilon = 1e-15
    ranked = tuple(
        tuple(sorted(enumerate(row), key=lambda item: (-item[1], item[0])))
        for row in prediction.probabilities
    )
    initial = (0, 0, 0, 0, 0)
    heap = [_heap_item(initial, ranked, epsilon)]
    visited = {initial}
    best = []
    while heap and len(best) < count:
        negative_log_probability, number_text, state, probabilities = heapq.heappop(heap)
        best.append((-negative_log_probability, number_text, probabilities))
        for position in range(5):
            if state[position] >= 9:
                continue
            neighbor = list(state)
            neighbor[position] += 1
            neighbor_state = tuple(neighbor)
            if neighbor_state in visited:
                continue
            visited.add(neighbor_state)
            heapq.heappush(heap, _heap_item(neighbor_state, ranked, epsilon))
    return [
        ProbabilityCandidate(
            rank=rank,
            number_text=number_text,
            model_name=prediction.model_name,
            probability=math.exp(log_probability),
            log_probability=log_probability,
            position_probabilities=probabilities,
            reason=_reason(number_text, probabilities),
        )
        for rank, (log_probability, number_text, probabilities) in enumerate(best, start=1)
    ]


def _heap_item(state, ranked, epsilon):
    choices = tuple(ranked[position][rank_index] for position, rank_index in enumerate(state))
    number_text = "".join(str(digit) for digit, _ in choices)
    probabilities = tuple(probability for _, probability in choices)
    log_probability = sum(math.log(max(value, epsilon)) for value in probabilities)
    return (-log_probability, number_text, state, probabilities)


def _reason(number_text: str, probabilities: Tuple[float, ...]) -> str:
    details = ", ".join(f"位置{index + 1}数字{digit}: {probability:.4f}" for index, (digit, probability) in enumerate(zip(number_text, probabilities)))
    return details
