from functools import total_ordering
from typing import List
from typing import cast as typing_cast

import numpy as np

from computational_models.models.abstract import AbstractLatticeModel, as_series, as_series_with


@total_ordering
class RealStateAgent:
    def __init__(
        self,
        agent_type: int,
        utility: float,
        capital: float = 1.0,
    ):
        self.agent_type = agent_type
        self.utility = utility
        self.capital = capital

    def __repr__(self) -> str:
        return str(self.agent_type)

    def __add__(self, delta_agent_type: int) -> int:
        if isinstance(delta_agent_type, type(self)):
            result = self.agent_type + delta_agent_type.agent_type
        else:
            result = self.agent_type + delta_agent_type
        return result

    def __sub__(self, delta_agent_type: int) -> int:
        if isinstance(delta_agent_type, type(self)):
            result = self.agent_type - delta_agent_type.agent_type
        else:
            result = self.agent_type - delta_agent_type
        return result

    def __mul__(self, factor: int) -> int:
        if isinstance(factor, type(self)):
            result = self.agent_type - factor.agent_type
        else:
            result = self.agent_type - factor
        return result

    def __eq__(self, other: object) -> bool:
        if isinstance(other, int):
            return self.agent_type == other
        elif isinstance(other, type(self)):
            return self.agent_type == other.agent_type
        else:
            return False

    def __hash__(self) -> int:
        return hash((self.agent_type,))

    def __lt__(self, other: object) -> bool:
        if isinstance(other, int):
            return self.agent_type < other
        elif isinstance(other, type(self)):
            return self.agent_type < other.agent_type
        else:
            return NotImplemented


class RealStateMarket(AbstractLatticeModel):
    def __init__(  # type: ignore[no-untyped-def]
        self,
        alpha: float = 0.5,
        A: float = 1 / 16,
        B: float = 0.5,
        utility_tolerance: float = 0.85,
        **kwargs,
    ):
        super(RealStateMarket, self).__init__(**kwargs)
        self.alpha = alpha
        self.A = A
        self.B = B
        self.utility_tolerance = utility_tolerance
        self.configuration = self._process_lattice_with(self.create_agent)

    def create_agent(self, i: int, j: int) -> RealStateAgent:
        similar_amount = self.similar_neighbors_amount(i, j, count_myself=True)
        initial_capital = 1.0
        property_price = self.property_price(similar_amount)
        return RealStateAgent(
            agent_type=self.get_agent(i, j),
            capital=initial_capital,
            utility=self.utility(initial_capital, property_price),
        )

    def get_real_state_agent(self, i: int, j: int) -> RealStateAgent:
        return typing_cast(RealStateAgent, self.get_agent(i, j))

    def property_price(self, similar_amount: int) -> float:
        distinct_amount = self.neighborhood.size() - similar_amount + 1
        return self.A * (similar_amount - distinct_amount) + self.B

    def utility(self, capital: float, price: float) -> float:
        return (capital ** (self.alpha)) * (price ** (1 - self.alpha))

    def swap(self) -> None:
        position_1, position_2 = self._random_positions_to_swap()
        agent_1 = self.get_real_state_agent(*position_1)
        agent_2 = self.get_real_state_agent(*position_2)
        target_similar_amount_1 = self.similar_neighbors_amount(
            *position_2,
            agent_type=agent_1.agent_type,
            count_myself=True,
        )
        target_similar_amount_2 = self.similar_neighbors_amount(
            *position_1,
            agent_type=agent_2.agent_type,
            count_myself=True,
        )

        target_new_price_1 = self.property_price(target_similar_amount_1)
        target_new_price_2 = self.property_price(target_similar_amount_2)
        average_price = (target_new_price_2 + target_new_price_1) / 2
        has_enough_capital_1 = target_new_price_1 - average_price - agent_1.capital < 0
        has_enough_capital_2 = target_new_price_2 - average_price - agent_2.capital < 0

        if has_enough_capital_1 and has_enough_capital_2:
            new_capital_1 = agent_1.capital + target_new_price_2 - average_price
            new_capital_2 = agent_2.capital + target_new_price_1 - average_price
            new_utility_1 = self.utility(new_capital_1, target_new_price_1)
            new_utility_2 = self.utility(new_capital_2, target_new_price_2)
            delta_utility_1 = new_utility_1 - agent_1.utility
            delta_utility_2 = new_utility_2 - agent_2.utility
            if delta_utility_1 > 0 and delta_utility_2 > 0:
                agent_1.capital = new_capital_1
                agent_2.capital = new_capital_2
                agent_1.utility = new_utility_1
                agent_2.utility = new_utility_2
                self.set_agent(*position_1, agent_2)
                self.set_agent(*position_2, agent_1)

    def utility_of(self, i: int, j: int) -> float:
        agent = self.get_real_state_agent(i, j)
        similar_amount = self.similar_neighbors_amount(i, j, count_myself=True)
        property_price = self.property_price(similar_amount)
        return self.utility(agent.capital, property_price)

    @as_series
    def agent_types_lattice(self, flatten: bool = False) -> List[List[int]]:
        action = lambda i, j: self.get_real_state_agent(i, j).agent_type
        return self._process_lattice_with(action, flatten=flatten)

    @as_series
    def utility_level_lattice(self, flatten: bool = False) -> List[List[int]]:
        return self._process_lattice_with(self.utility_of, flatten=flatten)

    @as_series
    def capital_level_lattice(self, flatten: bool = False) -> List[List[int]]:
        action = lambda i, j: self.get_real_state_agent(i, j).capital
        return self._process_lattice_with(action, flatten=flatten)

    @as_series_with(metadata={"states": ["satisfied", "dissatisfied"]})
    def dissatisfaction_threshold_lattice(self, flatten: bool = False) -> List[List[int]]:
        action = lambda i, j: self.utility_of(i, j) < self.utility_tolerance
        dissatisfaction_lattice = self._process_lattice_with(action, flatten=flatten)
        original_lattice = np.copy(self.configuration)
        return original_lattice + np.multiply(dissatisfaction_lattice, self.agent_types)

    @as_series_with(equilibrium_expected=True)
    def total_average_utility_level(self) -> float:
        total_utility = sum(self.utility_level_lattice(flatten=True))  # type: ignore[call-arg]
        return total_utility / self.length**2

    @as_series
    def total_average_capital_level(self) -> float:
        total_capital = sum(self.capital_level_lattice(flatten=True))  # type: ignore[call-arg]
        return total_capital / self.length**2
