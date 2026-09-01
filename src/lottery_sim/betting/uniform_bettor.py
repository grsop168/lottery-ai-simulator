from __future__ import annotations

from typing import Sequence

import numpy as np

from lottery_sim.betting.bet_distribution import PL5_NUMBER_COUNT
from lottery_sim.models import Draw5D


class UniformBettor:
    name = "uniform"

    def manual_probabilities(self, history: Sequence[Draw5D]) -> np.ndarray:
        return np.full(PL5_NUMBER_COUNT, 1.0 / PL5_NUMBER_COUNT, dtype=np.float64)
