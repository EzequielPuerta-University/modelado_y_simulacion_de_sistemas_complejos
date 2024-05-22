from src.models.abstract.agent import Agent


class ProfitExpectation:
    def __init__(
        self, initial: float, earn: float, within: int, delta: float, min_earnings: float = 0.05
    ) -> None:
        self.earn = earn
        self.target_profit = initial + (initial * earn)
        self.within_days = within
        self.delta_price = delta
        self.__initial_within_days = within
        self.__min_earnings = min_earnings
        self.__delta_earnings = 0.01

    def __repr__(self) -> str:
        txt = "{type(self).__name__} (earn={}, target_profit={}, within_days={}, delta_price={})"
        return txt.format(self.earn, self.target_profit, self.within_days, self.delta_price)

    def __restart_expectation(self, capital: float) -> None:
        self.target_profit = capital + (capital * self.earn)
        self.within_days = self.__initial_within_days

    def check(self, producer: "Producer") -> None:
        if self.within_days <= 0:
            price_decrement = producer.price * self.delta_price
            if producer.price > price_decrement:
                producer.price = producer.price - price_decrement
            if self.earn > self.__min_earnings:
                self.earn = self.earn - self.__delta_earnings
            self.__restart_expectation(producer.capital)
        elif producer.capital >= self.target_profit:
            producer.price = producer.price + (producer.price * self.delta_price)
            self.earn = self.earn + self.__delta_earnings
            self.__restart_expectation(producer.capital)
        else:
            # keep waiting...
            self.within_days = self.within_days - 1


class Producer(Agent):
    TYPE = 1

    def __init__(
        self,
        capital: float,
        stock: int,
        price: float,
        expectation: ProfitExpectation,
    ) -> None:
        self.capital = capital
        self.stock = stock
        self.price = price
        self.profit_expectation = expectation
        super(Producer, self).__init__(self.TYPE)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(capital={self.capital}, stock={self.stock}, price={self.price})"
        )

    def sale(self, amount: int = 1) -> None:
        if self.stock >= amount:
            self.stock = self.stock - amount
            self.capital = self.capital + (self.price * amount)
        else:
            raise AssertionError("Insufficient stock")

    def balance_check(self) -> None:
        self.profit_expectation.check(self)
