from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from lottery_sim.betting import BetDistribution, BetDistributionEstimator
from lottery_sim.betting.behavioral_bettor import BehavioralBettor
from lottery_sim.ml.base import ProbabilityPrediction
from lottery_sim.models import Draw5D


@dataclass(frozen=True)
class HouseNumberScore:
    number_text: str
    bet_probability: float
    expected_ticket_count: float
    expected_payout: float
    house_probability: float


class HouseMinPayoutModel:
    """Experimental minimum-payout hypothesis; this does not describe a proven draw mechanism."""

    model_name = "house"

    def __init__(
        self,
        estimator: Optional[BetDistributionEstimator] = None,
        house_temperature: float = 1_000_000.0,
        payout_per_ticket: float = 100_000.0,
    ):
        if house_temperature <= 0 and not math.isinf(house_temperature):
            raise ValueError("house_temperature must be positive or infinity")
        self.estimator = estimator or BetDistributionEstimator(BehavioralBettor())
        self.house_temperature = float(house_temperature)
        self.payout_per_ticket = float(payout_per_ticket)
        self.training_draw_count = 0
        self.last_market: BetDistribution | None = None
        self.last_expected_payout: np.ndarray | None = None
        self.last_house_probabilities: np.ndarray | None = None

    def fit(self, history: Sequence[Draw5D]) -> "HouseMinPayoutModel":
        self.training_draw_count = len(history)
        return self

    def predict_proba(self, history: Sequence[Draw5D]) -> ProbabilityPrediction:
        market = self.estimator.estimate(history)
        expected_payout = market.expected_ticket_counts * self.payout_per_ticket
        house_probabilities = _inverse_softmax(expected_payout, self.house_temperature)
        cube = house_probabilities.reshape((10, 10, 10, 10, 10))
        marginals = []
        for position in range(5):
            axes = tuple(axis for axis in range(5) if axis != position)
            marginals.append(tuple(float(value) for value in cube.sum(axis=axes)))
        self.last_market = market
        self.last_expected_payout = expected_payout
        self.last_house_probabilities = house_probabilities
        return ProbabilityPrediction(
            model_name=self.model_name,
            probabilities=tuple(marginals),
            train_draw_count=self.training_draw_count,
            metadata={
                "hypothesis": "minimum-payout",
                "bettor_model": market.bettor_model_name,
                "house_temperature": self.house_temperature,
                "simulated_ticket_count": market.simulated_ticket_count,
                "payout_per_ticket": self.payout_per_ticket,
            },
        )

    def number_score(self, number: int | str) -> HouseNumberScore:
        if self.last_market is None or self.last_expected_payout is None or self.last_house_probabilities is None:
            raise ValueError("predict_proba must be called before requesting full-number scores")
        index = int(number)
        if index < 0 or index >= 100_000:
            raise ValueError("PL5 number must be between 00000 and 99999")
        return HouseNumberScore(
            number_text=f"{index:05d}",
            bet_probability=float(self.last_market.probabilities[index]),
            expected_ticket_count=float(self.last_market.expected_ticket_counts[index]),
            expected_payout=float(self.last_expected_payout[index]),
            house_probability=float(self.last_house_probabilities[index]),
        )


def _inverse_softmax(expected_payout: np.ndarray, temperature: float) -> np.ndarray:
    if math.isinf(temperature):
        return np.full(expected_payout.shape, 1.0 / expected_payout.size, dtype=np.float64)
    logits = -np.asarray(expected_payout, dtype=np.float64) / temperature
    logits -= float(logits.max())
    weights = np.exp(logits)
    return weights / weights.sum()
