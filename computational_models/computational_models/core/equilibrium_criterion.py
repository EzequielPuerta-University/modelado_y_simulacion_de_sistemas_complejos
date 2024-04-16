from typing import List

import numpy as np


class EquilibriumCriterion:
    def __init__(
        self,
        window_size: int = 20,
        tolerance: float = 0.001,
    ):
        self.window_size: int = window_size
        self.tolerance: float = tolerance

    def in_equilibrium(self, series: List[float]) -> bool:
        length = len(series)
        if length <= self.window_size:
            return False
        else:
            window = range(length - self.window_size, length)
            return all(
                (
                    np.abs((series[i] - series[i - 1]) / series[i - 1]) < self.tolerance
                    for i in window
                )
            )
