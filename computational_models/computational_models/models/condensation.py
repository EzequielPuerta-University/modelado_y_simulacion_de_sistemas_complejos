from typing import List

import numpy as np

from computational_models.models.abstract import AbstractLatticeModel, as_series


class Condensation(AbstractLatticeModel):
    def __init__(self, probability: float, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.probability: float = probability
        super(Condensation, self).__init__(*args, **kwargs)

    def __configure_random_lattice(self) -> np.array:
        return (np.random.rand(self.length, self.length) < self.probability).astype(int)

    def __condensed_amount(self, i: int, j: int) -> int:
        return self.similar_neighbors_amount(i, j, agent_type=1)

    def step(self, i: int, j: int) -> None:
        agent_type = self.get_agent(i, j).agent_type
        neighbors = self.__condensed_amount(i, j) + agent_type
        if agent_type == 0 and neighbors >= 4:
            self.get_agent(i, j).agent_type = 1
        if agent_type == 1 and neighbors < 4:
            self.get_agent(i, j).agent_type = 0

    @as_series
    def agent_types_lattice(self, flatten: bool = False) -> List[List[int]]:
        action = lambda i, j: int(self.get_agent(i, j).agent_type)
        return self._process_lattice_with(action, flatten=flatten)
