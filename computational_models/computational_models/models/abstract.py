from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Tuple, Type

import numpy as np

from computational_models.core.equilibrium_criterion import EquilibriumCriterion
from computational_models.core.neighborhood import Neighborhood, VonNeumann


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
        self.configuration: np.ndarray = (
            configuration if configuration else self._generate_random_grid(agent_types)
        )
        self.agent_types: int = agent_types
        self.series: Dict[str, Any] = {}
        self.__configure_series()

    def _generate_random_grid(self, agent_types: int = 2) -> np.ndarray:
        return np.random.randint(agent_types, size=(self.length, self.length))

    def _random_positions_to_swap(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return (
            tuple(position)  # type: ignore[return-value]
            for position in np.random.randint(0, self.length, size=(2, 2))
        )

    def get_agent(self, i: int, j: int) -> int:
        return self.configuration[i][j]

    def set_agent(self, i: int, j: int, _with: Any) -> None:
        self.configuration[i][j] = _with

    def similar_neighbors_amount(
        self,
        i: int,
        j: int,
        agent_type: int | None = None,
        count_myself: bool = False,
    ) -> int:
        _agent_type = agent_type if agent_type else self.get_agent(i, j)
        like_minded_neighbors = [
            1
            for row, col in self.neighborhood.indexes_for(i, j)
            if self.get_agent(row, col) == _agent_type
        ]
        total = sum(like_minded_neighbors)
        return total + 1 if count_myself else total

    def _process_lattice_with(
        self, action: Callable[[int, int], Any], flatten: bool = False
    ) -> List[List[int]]:
        result = [[action(i, j) for j in range(self.length)] for i in range(self.length)]
        return sum(result, []) if flatten else result

    def __configure_series(self) -> None:
        for method_name in dir(self):
            method = getattr(self, method_name)
            try:
                if method.__is_series__:
                    self.series[method.__name__] = []
                    if method.__to_be_in_equilibrium__:
                        self.__equilibrium_series_name = method.__name__
            except AttributeError:
                pass

    def __save_series(self) -> None:
        for name, series in self.series.items():
            series.append(getattr(self, name)())

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
