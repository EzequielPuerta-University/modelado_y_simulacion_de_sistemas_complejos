import numpy as np
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
        _use_series_history = (
            len(runner.experiments) == 1
            and len(runner.experiments[0].series_history[series_name]) > 1
        )

        figure = go.Figure()
        for experiment in runner.experiments:
            if _use_series_history:
                series_history = experiment.series_history[series_name]
                series_collection = [
                    {"y": series, "name": f"# {_id}", "mode": "lines"}
                    for _id, series in enumerate(series_history)
                ]

                min_length = min(map(lambda each: len(each), series_history))
                series_history = [each[:min_length] for each in series_history]
                average = np.mean(np.array(series_history), axis=0, dtype="object")
                series_collection.append(
                    {
                        "y": average,
                        "name": "Average",
                        "line": {  # type: ignore[dict-item]
                            "color": "firebrick",
                            "width": 4,
                            "dash": "dot",
                        },
                    }
                )
                plot_title = f"{plot_title}<br>n = {len(series_history)}"
            else:
                params = runner.experiment_parameters_set.parameters_to_vary
                params = [f"{attribute}={getattr(experiment, attribute)}" for attribute in params]
                name = ", ".join(params)
                series = experiment.series[series_name]
                series_collection = [{"y": series, "name": name, "mode": "lines"}]
            for series in series_collection:
                figure.add_trace(go.Scatter(**series))

        figure.update_layout(
            title=plot_title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
        )
        figure.show()
