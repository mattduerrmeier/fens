import dataclasses
import pathlib
import typing


@dataclasses.dataclass
class VisualizationParameters:
    supports_visualization: bool
    results_path: pathlib.Path


def get_visualization_parameters(args: typing.Any) -> VisualizationParameters:
    supports_visualization = args.dataset == "MNIST"
    results_path = pathlib.Path(args.result_dir)

    return VisualizationParameters(supports_visualization, results_path)
