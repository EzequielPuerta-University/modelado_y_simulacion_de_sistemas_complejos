import pytest

from src.models.final_practice_work.market import Market
from src.simulation.core.equilibrium_criterion import WithoutCriterion
from src.simulation.core.experiment import ExperimentParametersSet
from src.simulation.core.neighborhood import Moore
from src.simulation.core.runner import Runner

experiment_parameters_set = ExperimentParametersSet(
    length=[50],
    neighborhood=[Moore],
    agent_types=[2],
    producer_probability=[0.65],
)
runner = Runner(
    Market,
    experiment_parameters_set,
    WithoutCriterion(),
    max_steps=5,
)


def test_market() -> None:
    assert len(runner.experiments) == 1
    with pytest.raises(AttributeError):
        runner.experiments[0].series

    runner.start()
    for series in runner.experiments[0].series.values():
        assert len(series) == 5 + 1
