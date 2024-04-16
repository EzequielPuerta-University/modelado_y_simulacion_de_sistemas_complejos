from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial, total_ordering
from typing import Any, Callable, Dict, List, Tuple, Type, cast

import numpy as np

from computational_models.core.equilibrium_criterion import EquilibriumCriterion
from computational_models.core.neighborhood import Neighborhood, VonNeumann


@total_ordering
class Agent(ABC):
    def __init__(
        self,
        agent_type: int,
    ):
        self.agent_type = agent_type

    def __repr__(self) -> str:
        return str(self.agent_type)

    def __add__(self, delta_agent_type: int | Agent) -> int:
        if isinstance(delta_agent_type, type(self)):
            result = self.agent_type + delta_agent_type.agent_type
        else:
            result = self.agent_type + cast(int, delta_agent_type)
        return result

    def __sub__(self, delta_agent_type: int | Agent) -> int:
        if isinstance(delta_agent_type, type(self)):
            result = self.agent_type - delta_agent_type.agent_type
        else:
            result = self.agent_type - cast(int, delta_agent_type)
        return result

    def __mul__(self, factor: int | Agent) -> int:
        if isinstance(factor, type(self)):
            result = self.agent_type * factor.agent_type
        else:
            result = self.agent_type * cast(int, factor)
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


class AbstractLatticeModel(ABC):
    def __init__(
        self,
        length: int,
        configuration: np.ndarray | None = None,
        neighborhood: Type[Neighborhood] = VonNeumann,
        agent_types: int = 2,
    ):
        self.length: int = length
        self.neighborhood: Neighborhood = neighborhood(self.length)
        self.agent_types: int = agent_types
        self.__configure_agents(configuration)
        self.__configure_series()

    def __configure_agents(self, configuration: np.ndarray | None) -> None:
        if configuration:
            _configuration = configuration
        else:
            _configuration = np.random.randint(self.agent_types, size=(self.length, self.length))

        create_agents = partial(self.__create_agent_as, self.__basic_agent, _configuration)
        self.configuration = self._process_lattice_with(create_agents)
        try:
            create_agents = partial(self.__create_agent_as, self._create_agent, self.configuration)
            self.configuration = self._process_lattice_with(create_agents)
        except NotImplementedError:
            pass

    def __create_agent_as(
        self,
        method: Callable[[int, int, int], Agent],
        _configuration: np.ndarray,
        i: int,
        j: int,
    ) -> Agent:
        return method(_configuration[i][j], i, j)

    def __basic_agent(self, agent_type: int, i: int, j: int) -> Agent:
        return Agent(agent_type=agent_type)

    def _create_agent(self, basic_agent: Agent, i: int, j: int) -> Agent:
        # Overload this method in your model to create custom agents based on
        # basic agents previously created, in the (i,j) position of the
        # configuration lattice, replacing the older basic model.
        raise NotImplementedError

    def _process_lattice_with(
        self, action: Callable[[int, int], Any], flatten: bool = False
    ) -> List[List[Any]]:
        result = [[action(i, j) for j in range(self.length)] for i in range(self.length)]
        return sum(result, []) if flatten else result

    def __configure_series(self) -> None:
        self.series: Dict[str, Any] = {}
        for method_name in dir(self):
            method = getattr(self, method_name)
            try:
                if method.__is_series__:
                    self.series[method.__name__] = []
                    if method.__to_be_in_equilibrium__:
                        self.__equilibrium_series_name = method.__name__
            except AttributeError:
                pass

    def _random_positions_to_swap(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return (
            tuple(position)  # type: ignore[return-value]
            for position in np.random.randint(0, self.length, size=(2, 2))
        )

    def get_agent(self, i: int, j: int) -> Agent:
        return self.configuration[i][j]

    def set_agent(self, i: int, j: int, _with: Agent) -> None:
        self.configuration[i][j] = _with

    def similar_neighbors_amount(
        self,
        i: int,
        j: int,
        agent_type: int | None = None,
        count_myself: bool = False,
    ) -> int:
        _agent_type = agent_type if agent_type else self.get_agent(i, j).agent_type
        like_minded_neighbors = [
            1
            for row, col in self.neighborhood.indexes_for(i, j)
            if self.get_agent(row, col).agent_type == _agent_type
        ]
        total = sum(like_minded_neighbors)
        return total + 1 if count_myself else total

    def run_with(
        self,
        max_steps: int,
        criterion: EquilibriumCriterion,
    ) -> None:
        self.__save_series()
        for _ in range(max_steps):
            self.run_step()
            self.__save_series()
            if criterion.in_equilibrium(self.series[self.__equilibrium_series_name]):
                break

    def __save_series(self) -> None:
        for name, series in self.series.items():
            series.append(getattr(self, name)())

    def run_step(self) -> None:
        for _ in range(self.length**2):
            self.swap()

    @abstractmethod
    def swap(self) -> None:
        pass


def __as_series(
    model_function: Callable[[Any], Any],
    equilibrium_expected: bool = False,
    metadata: Dict[str, Any] | None = None,
) -> Callable[[Any], Any]:
    model_function.__is_series__ = True  # type: ignore[attr-defined]
    model_function.__to_be_in_equilibrium__ = equilibrium_expected  # type: ignore[attr-defined]
    model_function.__series_metadata__ = metadata if metadata else {}  # type: ignore[attr-defined]
    return model_function


def as_series(model_function: Callable[[Any], Any]) -> Callable[[Any], Any]:
    return __as_series(model_function)


def as_series_with(
    equilibrium_expected: bool = False,
    metadata: Dict[str, Any] | None = None,
) -> Callable[[Any], Any]:
    def decorator(model_function: Callable[[Any], Any]) -> Callable[[Any], Any]:
        return __as_series(model_function, equilibrium_expected, metadata)

    return decorator
