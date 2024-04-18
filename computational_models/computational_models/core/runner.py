from typing import List, Tuple, Type

from computational_models.core.equilibrium_criterion import EquilibriumCriterion
from computational_models.core.experiment import ExperimentParametersSet
from computational_models.models.abstract import AbstractLatticeModel


class Execute:
    def __init__(self, *series_names: Tuple[str, ...], times: int = 1) -> None:
        self.series_names: Tuple[str] = series_names  # type: ignore[assignment]
        self.times: int = times


class Runner:
    def __init__(
        self,
        model: Type[AbstractLatticeModel],
        experiment_parameters_set: ExperimentParametersSet,
        equilibrium_criterion: EquilibriumCriterion,
        max_steps: int = 150,
        repeat: Execute = Execute(),
    ):
        if repeat.times > 1:
            assert all(
                (len(each) == 1 for each in experiment_parameters_set._raw.values())
            ), """Repetition mode only supports one experiment per runner. \
                Please, reduce your Parameters Set."""
        self.experiments: List[AbstractLatticeModel] = []
        self.equilibrium_criterion = equilibrium_criterion
        self.max_steps = max_steps
        self.repeat = repeat
        self.experiment_parameters_set = experiment_parameters_set
        try:
            for experiment_parameters in experiment_parameters_set:
                experiment = model(**experiment_parameters)
                self.experiments.append(experiment)
        except TypeError as error:
            raise TypeError(
                f"{error}. "
                f"Check the attributes in the {type(experiment_parameters_set).__name__} instance. "
                f"They should be named equal to the names expected by the {model.__name__} model."
            )

    def start(self) -> None:
        for _ in range(self.repeat.times):
            for experiment in self.experiments:
                experiment.run_with(
                    max_steps=self.max_steps,
                    criterion=self.equilibrium_criterion,
                    saving_series=self.repeat.series_names,
                )
