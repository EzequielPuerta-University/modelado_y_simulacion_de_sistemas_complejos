from typing import Callable

import pytest

from src.models.final_practice_work.producer import Producer, ProfitExpectation


@pytest.fixture
def initial_capital() -> float:  # type: ignore[misc]
    yield 100


@pytest.fixture
def profit_expectation(initial_capital: float) -> Callable:  # type: ignore
    def __profit_expectation(
        earn: float = 0.1,
        within: int = 5,
        delta: float = 0.05,
    ) -> ProfitExpectation:
        return ProfitExpectation(
            initial_capital,
            earn,
            within,
            delta,
        )

    yield __profit_expectation


@pytest.fixture
def producer(initial_capital) -> Callable:  # type: ignore
    def __producer(
        expectation: ProfitExpectation,
        stock: int = 20,
        price: float = 2,
    ) -> Producer:
        return Producer(
            capital=initial_capital,
            stock=stock,
            price=price,
            expectation=expectation,
        )

    yield __producer


def test_producer_creation(  # type: ignore[no-untyped-def]
    producer,
    profit_expectation,
) -> None:
    producer = producer(expectation=profit_expectation)
    assert producer.capital == 100
    assert producer.stock == 20
    assert producer.price == 2
    assert producer.TYPE == 1
    assert producer.agent_type == producer.TYPE


def test_producer_sales(  # type: ignore[no-untyped-def]
    producer,
    profit_expectation,
) -> None:
    producer = producer(expectation=profit_expectation)
    assert producer.stock == 20
    assert producer.capital == 100
    producer.sale()
    assert producer.stock == 19
    assert producer.capital == 102


def test_producer_with_insufficient_stock(  # type: ignore[no-untyped-def]
    producer,
    profit_expectation,
) -> None:
    producer = producer(expectation=profit_expectation)
    assert producer.stock == 20
    assert producer.capital == 100
    with pytest.raises(AssertionError) as error:
        producer.sale(amount=21)
    assert error.value.args[0] == "Insufficient stock"
    assert producer.stock == 20
    assert producer.capital == 100


def test_producer_increases_the_price(  # type: ignore[no-untyped-def]
    producer,
    profit_expectation,
) -> None:
    days = 5
    profit_percentage = 0.1
    delta_price = 0.05
    profit_expectation = profit_expectation(
        earn=profit_percentage,
        within=days,
        delta=delta_price,
    )
    producer = producer(expectation=profit_expectation)
    initial_capital = producer.capital

    for _ in range(days):
        assert producer.price == 2
        assert producer.capital < initial_capital + (initial_capital * profit_percentage)
        producer.sale()
        producer.balance_check()
    assert producer.price == 2.1
    assert producer.capital >= initial_capital + (initial_capital * profit_percentage)


def test_producer_decreases_the_price(  # type: ignore[no-untyped-def]
    producer,
    profit_expectation,
) -> None:
    days = 5
    profit_percentage = 0.1
    delta_price = 0.05
    profit_expectation = profit_expectation(
        earn=profit_percentage,
        within=days,
        delta=delta_price,
    )
    producer = producer(expectation=profit_expectation)
    initial_capital = producer.capital

    for day in range(days + 1):
        assert producer.price == 2
        assert producer.capital < initial_capital + (initial_capital * profit_percentage)
        if day % 2 == 0:
            producer.sale()
        producer.balance_check()
    assert producer.price == 1.9
    assert producer.capital < initial_capital + (initial_capital * profit_percentage)


def test_producer_can_increase_the_price_when_target_is_reached(  # type: ignore[no-untyped-def]
    producer,
    profit_expectation,
) -> None:
    days = 5
    profit_percentage = 0.1
    delta_price = 0.05
    profit_expectation = profit_expectation(
        earn=profit_percentage,
        within=days,
        delta=delta_price,
    )
    producer = producer(expectation=profit_expectation)
    initial_capital = producer.capital

    for _ in range(days - 2):
        assert producer.price == 2
        assert producer.capital < initial_capital + (initial_capital * profit_percentage)
        producer.sale(amount=2)
        producer.balance_check()
    assert producer.price == 2.1
    assert producer.capital >= initial_capital + (initial_capital * profit_percentage)


def test_producer_can_increase_the_price_many_times(  # type: ignore[no-untyped-def]
    producer,
    profit_expectation,
) -> None:
    days = 5
    profit_percentage = 0.1
    delta_price = 0.05
    profit_expectation = profit_expectation(
        earn=profit_percentage,
        within=days,
        delta=delta_price,
    )
    producer = producer(expectation=profit_expectation)
    initial_capital = producer.capital

    for _ in range(days):
        assert producer.price == 2
        assert producer.capital < initial_capital + (initial_capital * profit_percentage)
        producer.sale(amount=1)
        producer.balance_check()
    assert producer.price == 2.1
    assert producer.capital >= initial_capital + (initial_capital * profit_percentage)

    initial_capital = producer.capital
    for _ in range(days - 2):
        assert producer.price == 2.1
        assert producer.capital < initial_capital + (initial_capital * profit_percentage)
        producer.sale(amount=2)
        producer.balance_check()
    assert producer.price == 2.205
    assert producer.capital >= initial_capital + (initial_capital * profit_percentage)


def test_producer_can_decrease_the_price_many_times(  # type: ignore[no-untyped-def]
    producer,
    profit_expectation,
) -> None:
    days = 5
    profit_percentage = 0.1
    delta_price = 0.05
    profit_expectation = profit_expectation(
        earn=profit_percentage,
        within=days,
        delta=delta_price,
    )
    producer = producer(expectation=profit_expectation)
    initial_capital = producer.capital

    for day in range(days + 1):
        assert producer.price == 2
        assert producer.capital < initial_capital + (initial_capital * profit_percentage)
        if day % 2 == 0:
            producer.sale()
        producer.balance_check()
    assert producer.price == 1.9
    assert producer.capital < initial_capital + (initial_capital * profit_percentage)

    initial_capital = producer.capital
    for day in range(days + 1):
        assert producer.price == 1.9
        assert producer.capital < initial_capital + (initial_capital * profit_percentage)
        if day % 2 == 0:
            producer.sale()
        producer.balance_check()
    assert producer.price == 1.805
    assert producer.capital < initial_capital + (initial_capital * profit_percentage)
