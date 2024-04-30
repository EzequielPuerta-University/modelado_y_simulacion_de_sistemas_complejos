from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, List, Tuple, Type, Union

import numpy as np

from src.models.abstract.agent import Agent
from src.simulation.core.equilibrium_criterion import EquilibriumCriterion
from src.simulation.core.neighborhood import Neighborhood, VonNeumann


class AbstractLatticeModel(ABC):
    def __init__(
        self,
        length: int,
        configuration: np.ndarray | None = None,
        neighborhood: Type[Neighborhood] = VonNeumann,
        agent_types: int = 2,
        update_simultaneously: bool = False,
    ):
        self.length: int = length
        self.neighborhood: Neighborhood = neighborhood(self.length)
        self.agent_types: int = agent_types
        self.update_simultaneously: bool = update_simultaneously
        self.series_history: Dict[str, List[List[Union[int, float]]]] = {}
        self.__initial_configuration: np.ndarray | None = configuration

    def __initialize(self) -> None:
        self.__configure_agents()
        self.__configure_series()

    def __configure_agents(self) -> None:
        if self.__initial_configuration is not None:
            raw_configuration = self.__initial_configuration
        else:
            raw_configuration = self._configure_random_lattice()

        create_agents = partial(self.__create_agent_as, self.__basic_agent, raw_configuration)
        self.configuration = self._process_lattice_with(create_agents)
        try:
            create_agents = partial(self.__create_agent_as, self._create_agent, self.configuration)
            self.configuration = self._process_lattice_with(create_agents)
        except NotImplementedError:
            pass

    def _configure_random_lattice(self) -> np.array:
        return np.random.randint(self.agent_types, size=(self.length, self.length))

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
            except AttributeError:
                pass

    def __take_snapshot(self) -> None:
        for name, series in self.series.items():
            series.append(getattr(self, name)())

    def __save_series_history(self, series: Tuple[str]) -> None:
        if len(series) > 0:
            for name in series:
                try:
                    history = self.series_history[name]
                except KeyError:
                    self.series_history[name] = []
                    history = self.series_history[name]
                finally:
                    try:
                        history.append(self.series[name])
                    except KeyError:
                        raise ValueError(f"There is no series named as '{name}'.")

    def _random_positions_to_swap(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return (
            tuple(position)  # type: ignore[return-value]
            for position in np.random.randint(0, self.length, size=(2, 2))
        )

    def get_agent(self, i: int, j: int) -> Agent:
        return self.configuration[i][j]

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
        saving_series: Tuple[str],
    ) -> None:
        self.__initialize()
        self.__take_snapshot()
        for _ in range(max_steps):
            self.run_step()
            self.__take_snapshot()
            if criterion.in_equilibrium(self.series):
                break
        self.__save_series_history(series=saving_series)

    def run_step(self) -> None:
        configuration = (
            deepcopy(self.configuration) if self.update_simultaneously else self.configuration
        )
        for i, j in ((i, j) for i in range(self.length) for j in range(self.length)):
            self.step(i, j, configuration=configuration)
        if self.update_simultaneously:
            self.configuration = configuration

    @abstractmethod
    def step(
        self,
        i: int,
        j: int,
        configuration: np.ndarray,
    ) -> None:
        pass


def __as_series(
    model_function: Callable[[Any], Any],
    metadata: Dict[str, Any] | None = None,
) -> Callable[[Any], Any]:
    model_function.__is_series__ = True  # type: ignore[attr-defined]
    model_function.__series_metadata__ = metadata if metadata else {}  # type: ignore[attr-defined]
    return model_function


def as_series(model_function: Callable[[Any], Any]) -> Callable[[Any], Any]:
    return __as_series(model_function)


def as_series_with(
    metadata: Dict[str, Any] | None = None,
) -> Callable[[Any], Any]:
    def decorator(model_function: Callable[[Any], Any]) -> Callable[[Any], Any]:
        return __as_series(model_function, metadata)

    return decorator
