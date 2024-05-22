from typing import Callable, List, cast

from src.models.abstract.agent import Agent
from src.models.abstract.model import AbstractLatticeModel, as_series, as_series_with
from src.models.final_practice_work.consumer import Consumer
from src.models.final_practice_work.producer import Producer, ProfitExpectation
from src.simulation.core.lattice import Lattice


class Market(AbstractLatticeModel):
    def __init__(  # type: ignore[no-untyped-def]
        self,
        capital: float = 1000000,
        stock: int = 5000,
        price: float = 100,
        earn: float = 0.1,
        within_days: int = 10,
        delta_price: float = 0.015,
        min_earnings: float = 0.05,
        producer_probability: float = 0.1,
        *args,
        **kwargs,
    ):
        self.capital = capital
        self.stock = stock
        self.price = price
        self.earn = earn
        self.within_days = within_days
        self.delta_price = delta_price
        self.min_earnings = min_earnings
        self.producer_probability = producer_probability

        length = kwargs.get("length")
        configuration = kwargs.get(
            "configuration",
            Lattice.with_probability(
                self.producer_probability,
                cast(int, length),
            ),
        )

        super(Market, self).__init__(
            *args,
            update_simultaneously=True,
            update_sorted_by_agent_type=True,
            configuration=configuration,  # type: ignore[misc]
            **kwargs,
        )

    def _create_agent(self, basic_agent: Agent, i: int, j: int) -> Agent:
        if basic_agent.agent_type == Consumer.TYPE:
            agent = Consumer()
            self._by_type[Consumer.TYPE].append((i, j))
        elif basic_agent.agent_type == Producer.TYPE:
            agent = Producer(  # type: ignore[assignment]
                capital=self.capital,
                stock=self.stock,
                price=self.price,
                expectation=ProfitExpectation(
                    initial=self.capital,
                    earn=self.earn,
                    within=self.within_days,
                    delta=self.delta_price,
                    min_earnings=self.min_earnings,
                ),
            )
            self._by_type[Producer.TYPE].append((i, j))
        else:
            raise ValueError(
                f"Invalid agent type. Values {Consumer.TYPE} or {Producer.TYPE} expected"
            )
        return agent

    def __sellers_for(
        self,
        i: int,
        j: int,
        configuration: Lattice,
    ) -> List[Producer]:
        neighbors = self.neighborhood.indexes_for(i, j)
        sellers = []
        for position in neighbors:
            agent = configuration.at(*position)
            if agent.agent_type == Producer.TYPE:
                sellers.append(agent)
        return sellers

    def step(
        self,
        i: int,
        j: int,
        configuration: Lattice,
    ) -> None:
        agent = configuration.at(i, j)
        _type = agent.agent_type
        if _type == Consumer.TYPE:
            agent.buy(sellers=self.__sellers_for(i, j, configuration))
        elif _type == Producer.TYPE:
            agent.balance_check()
        else:
            raise ValueError(f"Unexpected agent type {_type} at ({i}, {j})")

    @as_series
    def agent_types_lattice(self) -> List[List[int]]:
        action = lambda i, j: int(self.get_agent(i, j).agent_type)
        return self._process_lattice_with(action)

    @as_series
    def price_lattice(self) -> List[List[float]]:
        action = lambda i, j: int(self.get_agent(i, j).price)  # type: ignore[attr-defined]
        return self._process_lattice_with(action)

    @as_series_with(depends=("price_lattice",))
    def average_price(self) -> float:
        prices = self._flatten("price_lattice")
        return sum(prices) / self.length**2

    def __collect(self, agent_type: int) -> Callable:  # type: ignore[type-arg]
        def _collector(i: int, j: int) -> float | None:
            agent = self.get_agent(i, j)
            if agent.agent_type == agent_type:
                return agent.price  # type: ignore[attr-defined]
            else:
                return None

        return _collector

    @as_series
    def average_consumer_price(self) -> float:
        prices = self._process_lattice_with(
            self.__collect(Consumer.TYPE),
            flatten=True,
        )
        prices = filter(lambda price: price is not None, prices)  # type: ignore[assignment]
        return sum(prices) / len(self._by_type[Consumer.TYPE])  # type: ignore[arg-type]

    @as_series
    def average_producer_price(self) -> float:
        prices = self._process_lattice_with(
            self.__collect(Producer.TYPE),
            flatten=True,
        )
        prices = filter(lambda price: price is not None, prices)  # type: ignore[assignment]
        return sum(prices) / len(self._by_type[Producer.TYPE])  # type: ignore[arg-type]
