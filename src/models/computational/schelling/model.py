from typing import List

import numpy as np

from src.models.abstract.model import AbstractLatticeModel, as_series, as_series_with


class Schelling(AbstractLatticeModel):
    def __init__(  # type: ignore[no-untyped-def]
        self,
        tolerance: int,
        *args,
        **kwargs,
    ):
        super(Schelling, self).__init__(*args, **kwargs)
        assert 1 < tolerance <= self.neighborhood.size(), (
            f"Tolerance threshold should be in range " f"(1, {self.neighborhood.size()}]"
        )
        self.tolerance: int = tolerance

    def step(
        self,
        i: int,
        j: int,
        configuration: np.ndarray,
    ) -> None:
        position_1, position_2 = self._random_positions_to_swap()
        if self.get_agent(*position_1) != self.get_agent(*position_2):
            conditions = (
                self.similar_neighbors_amount(*each) < self.tolerance
                for each in [position_1, position_2]
            )
            if all(conditions):
                temp = self.get_agent(*position_1)
                configuration[position_1[0]][position_1[1]] = self.get_agent(*position_2)
                configuration[position_2[0]][position_2[1]] = temp

    @as_series
    def agent_types_lattice(self) -> List[List[int]]:
        action = lambda i, j: self.get_agent(i, j).agent_type
        return self._process_lattice_with(action)

    @as_series
    def satisfaction_level_lattice(self, flatten: bool = False) -> List[List[int]]:
        return self._process_lattice_with(self.similar_neighbors_amount, flatten=flatten)

    @as_series_with(metadata={"states": ["satisfied", "dissatisfied"]})
    def dissatisfaction_threshold_lattice(self) -> List[List[int]]:
        action = lambda i, j: (
            self.get_agent(i, j).agent_type + self.agent_types
            if self.similar_neighbors_amount(i, j) < self.tolerance
            else self.get_agent(i, j).agent_type
        )
        return self._process_lattice_with(action)

    @as_series
    def total_average_satisfaction_level(self) -> float:
        satisfaction = sum(self.satisfaction_level_lattice(flatten=True))  # type: ignore[call-arg]
        return satisfaction / self.length**2
