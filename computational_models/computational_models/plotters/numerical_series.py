import plotly.graph_objects as go

from computational_models.core.runner import Runner


class NumericalSeries:
    @classmethod
    def show_up(
        cls,
        series_name: str,
        runner: Runner,
        plot_title: str,
        yaxis_title: str,
        xaxis_title: str = "# Steps",
        height: int | None = None,
        leyend: str = "",
    ) -> None:
        params = runner.experiment_parameters_set.parameters_to_vary

        figure = go.Figure()
        for experiment in runner.experiments:
            series = experiment.series[series_name]
            name = ", ".join(
                [f"{attribute}={getattr(experiment, attribute)}" for attribute in params]
            )
            figure.add_trace(go.Scatter(y=series, mode="lines", name=name))

        figure.update_layout(title=plot_title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
        figure.show()
