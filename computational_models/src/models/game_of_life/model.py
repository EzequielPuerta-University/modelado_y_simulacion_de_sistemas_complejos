from typing import List, cast

import numpy as np

from computational_models.src.models.abstract.model import AbstractLatticeModel, as_series
from computational_models.src.models.game_of_life.seeds import Seed


class GameOfLife(AbstractLatticeModel):
    ALIVE = 1
    DEAD = 0

    def __init__(self, seeds: List[Seed], *args, **kwargs):  # type: ignore[no-untyped-def]
        length = kwargs.get("length")
        _configuration = np.zeros((length, length))
        self.seeds = seeds
        for seed in self.seeds:
            seed._apply_on(_configuration)

        super(GameOfLife, self).__init__(
            *args,
            configuration=_configuration,  # type: ignore[misc]
            update_simultaneously=True,
            **kwargs,
        )

    def step(self, i: int, j: int, **kwargs) -> None:  # type: ignore[no-untyped-def]
        new_configuration = cast(np.ndarray, kwargs.get("new_configuration"))
        amount = self.similar_neighbors_amount(i, j, agent_type=self.ALIVE)
        if self.get_agent(i, j).agent_type == self.ALIVE:
            if amount in [2, 3]:
                new_state = self.ALIVE
            else:
                new_state = self.DEAD
        else:
            if amount == 3:
                new_state = self.ALIVE
            else:
                new_state = self.DEAD
        new_configuration[i][j].agent_type = new_state

    @as_series
    def agent_types_lattice(self, flatten: bool = False) -> List[List[int]]:
        action = lambda i, j: self.get_agent(i, j).agent_type
        return self._process_lattice_with(action, flatten=flatten)
