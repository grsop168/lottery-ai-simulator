from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from lottery_sim.betting.base import BettorModel
from lottery_sim.models import Draw5D


PL5_NUMBER_COUNT = 100_000


@dataclass(frozen=True)
class BetDistribution:
    probabilities: np.ndarray
    expected_ticket_counts: np.ndarray
    simulated_ticket_count: int
    bettor_model_name: str

    def __post_init__(self) -> None:
        probabilities = np.asarray(self.probabilities, dtype=np.float64)
        tickets = np.asarray(self.expected_ticket_counts, dtype=np.float64)
        if probabilities.shape != (PL5_NUMBER_COUNT,) or tickets.shape != (PL5_NUMBER_COUNT,):
            raise ValueError("PL5 betting distribution must contain all 100000 numbers")
        if np.any(probabilities < 0) or not np.isclose(float(probabilities.sum()), 1.0):
            raise ValueError("bet probabilities must be non-negative and sum to one")
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "expected_ticket_counts", tickets)


class BetDistributionEstimator:
    def __init__(
        self,
        bettor_model: BettorModel,
        simulated_ticket_count: int = 1_000_000,
        random_seed: int = 20260505,
        manual_pick_ratio: float = 0.60,
        machine_pick_ratio: float = 0.40,
    ):
        if simulated_ticket_count <= 0:
            raise ValueError("simulated_ticket_count must be positive")
        if manual_pick_ratio < 0 or machine_pick_ratio < 0:
            raise ValueError("pick ratios cannot be negative")
        total_ratio = manual_pick_ratio + machine_pick_ratio
        if total_ratio <= 0:
            raise ValueError("at least one pick ratio must be positive")
        self.bettor_model = bettor_model
        self.simulated_ticket_count = int(simulated_ticket_count)
        self.random_seed = int(random_seed)
        self.manual_pick_ratio = float(manual_pick_ratio) / total_ratio
        self.machine_pick_ratio = float(machine_pick_ratio) / total_ratio

    def estimate(self, history: Sequence[Draw5D]) -> BetDistribution:
        manual = np.asarray(self.bettor_model.manual_probabilities(history), dtype=np.float64)
        if manual.shape != (PL5_NUMBER_COUNT,):
            raise ValueError("bettor model must return all 100000 PL5 probabilities")
        uniform = np.full(PL5_NUMBER_COUNT, 1.0 / PL5_NUMBER_COUNT, dtype=np.float64)
        probabilities = self.manual_pick_ratio * manual + self.machine_pick_ratio * uniform
        probabilities /= probabilities.sum()
        return BetDistribution(
            probabilities=probabilities,
            expected_ticket_counts=probabilities * self.simulated_ticket_count,
            simulated_ticket_count=self.simulated_ticket_count,
            bettor_model_name=self.bettor_model.name,
        )
