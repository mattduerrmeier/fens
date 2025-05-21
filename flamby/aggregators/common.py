import typing


class AggregatorResult(typing.TypedDict):
    mse_loss: float
    downstream_train_accuracy: float
    downstream_test_accuracy: float
