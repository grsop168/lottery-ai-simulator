from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from lottery_sim.models import Draw5D


class BettorModel(Protocol):
    name: str

    def manual_probabilities(self, history: Sequence[Draw5D]) -> np.ndarray:
        """Return normalized estimated manual-pick probabilities for 00000..99999."""
        ...
