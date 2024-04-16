from typing import Any, List

from computational_models.core.equilibrium_criterion import EquilibriumCriterion
from computational_models.core.experiment import ExperimentParametersSet


class Runner:
    def __init__(
        self,
        model: Any,
        experiment_parameters_set: ExperimentParametersSet,
        equilibrium_criterion: EquilibriumCriterion,
        max_steps: int = 150,
    ):
        self.experiments: List[Any] = []
        self.equilibrium_criterion = equilibrium_criterion
        self.max_steps = max_steps
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
        for experiment in self.experiments:
            experiment.run_with(
                max_steps=self.max_steps,
                criterion=self.equilibrium_criterion,
            )
