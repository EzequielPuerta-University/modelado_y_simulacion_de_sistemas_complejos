from typing import List

from src.models.abstract.agent import Agent
from src.models.final_practice_work.producer import Producer


class Consumer(Agent):
    TYPE = 0

    def __init__(self) -> None:
        self.price: float = 0
        super(Consumer, self).__init__(self.TYPE)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(price={self.price})"

    def buy(self, sellers: List[Producer], amount: int = 1) -> None:
        cheapest = sellers[0]
        for seller in sellers[1:]:
            if seller.price < cheapest.price:
                cheapest = seller
        cheapest.sale(amount)
        self.price = cheapest.price
