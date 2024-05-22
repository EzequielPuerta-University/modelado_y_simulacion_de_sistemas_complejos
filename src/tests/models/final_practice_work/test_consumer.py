from typing import Callable

import pytest

from src.models.final_practice_work.consumer import Consumer
from src.models.final_practice_work.producer import Producer, ProfitExpectation


@pytest.fixture
def producer() -> Callable:  # type: ignore
    expectation = ProfitExpectation(
        initial=100,
        earn=0.1,
        within=5,
        delta=0.05,
    )

    def __producer(price) -> Producer:  # type: ignore[no-untyped-def]
        return Producer(
            capital=100,
            stock=10,
            price=price,
            expectation=expectation,
        )

    yield __producer


def test_consumer_creation() -> None:
    consumer = Consumer()
    assert consumer.TYPE == 0
    assert consumer.agent_type == consumer.TYPE
    assert consumer.price == 0


def test_consumer_buys_the_cheapest_product(  # type: ignore[no-untyped-def]
    producer,
) -> None:
    capital = 100
    stock = 10
    cheapest = producer(1.90)
    not_bad = producer(1.95)
    regular = producer(2.00)
    expensive = producer(2.05)
    not_cheap = [not_bad, expensive, regular]
    sellers = not_cheap + [cheapest]
    for seller in sellers:
        assert seller.capital == capital
        assert seller.stock == stock

    consumer = Consumer()
    assert consumer.price == 0
    amount_to_buy = 1
    consumer.buy(sellers=sellers, amount=amount_to_buy)

    assert consumer.price == cheapest.price
    for seller in not_cheap:
        assert seller.capital == capital
        assert seller.stock == stock
    assert cheapest.capital == capital + (cheapest.price * amount_to_buy)
    assert cheapest.stock == stock - amount_to_buy
