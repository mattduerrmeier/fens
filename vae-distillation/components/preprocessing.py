import typing
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_features: typing.Mapping[str, typing.Literal["categorical", "numerical"]] = {
    "age": "numerical",
    "sex": "categorical",
    "cp": "categorical",
    "trestbps": "numerical",
    "chol": "numerical",
    "fbs": "categorical",
    "restecg": "categorical",
    "thalach": "numerical",
    "exang": "categorical",
    "oldpeak": "numerical",
    "slope": "categorical",
    "ca": "categorical",
    "thal": "categorical",
}

feature_names = list(_features.keys())


def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=np.float32)


class EncodingFunction(typing.Protocol):
    def __call__(self, dataset: pd.DataFrame) -> pd.DataFrame: ...


def encode_dataset(
    dataset: pd.DataFrame,
    *,
    categorical_encoder: EncodingFunction,
    numerical_encoder: EncodingFunction,
) -> pd.DataFrame:
    encoded_frames: list[pd.DataFrame] = []

    categorical_features = [
        feature_name
        for feature_name in dataset.columns
        if _features.get(feature_name) == "categorical"
    ]
    if len(categorical_features) > 0:
        encoded_categorical_dataset = categorical_encoder(dataset[categorical_features])
        encoded_frames.append(encoded_categorical_dataset)

    numerical_features = [
        feature_name
        for feature_name in dataset.columns
        if _features.get(feature_name) == "numerical"
    ]
    if len(numerical_features) > 0:
        encoded_numerical_dataset = numerical_encoder(dataset[numerical_features])
        encoded_frames.append(encoded_numerical_dataset)

    if "target" in dataset.columns:
        encoded_frames.append(dataset["target"].to_frame())

    return pd.concat(encoded_frames, axis=1)


def as_tensor(dataset: pd.DataFrame, columns: list[str] | None = None) -> torch.Tensor:
    if columns is None:
        columns = list(dataset.columns)

    return torch.tensor(dataset[columns].values)


def partition_frame(
    frame: pd.DataFrame, split: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not (0 <= split <= 1):
        raise ValueError(f"expect split to be a fraction of one, got {split}")

    pivot_element_idx = int(split * len(frame))

    return (
        frame.loc[:pivot_element_idx],
        frame.loc[pivot_element_idx:],
    )


def partition_series(series: pd.Series, split: float) -> tuple[pd.Series, pd.Series]:
    if not (0 <= split <= 1):
        raise ValueError(f"expect split to be a fraction of one, got {split}")

    pivot_element_idx = int(split * len(series))

    return (
        series.loc[:pivot_element_idx],
        series.loc[pivot_element_idx:],
    )


def build_client_datasets(
    dataset: pd.DataFrame,
    train_test_split: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = dataset.drop("target", axis=1)
    return partition_frame(dataset, train_test_split)


def build_target_dataset(
    dataset: pd.DataFrame,
    train_test_split: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_train, target_test = partition_frame(
        dataset["target"].to_frame(), train_test_split
    )

    return target_train, target_test
