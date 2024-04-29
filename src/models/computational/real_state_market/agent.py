from src.models.abstract.agent import Agent


class RealStateAgent(Agent):
    def __init__(
        self,
        agent_type: int,
        utility: float,
        capital: float = 1.0,
    ):
        super(RealStateAgent, self).__init__(agent_type)
        self.utility = utility
        self.capital = capital
