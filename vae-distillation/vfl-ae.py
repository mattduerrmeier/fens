from pathlib import Path

import numpy as np
import pandas as pd
import torch
from evaluation import HeartDiseaseNN as EvaluatorModel
from components import encoders, preprocessing
from models.autoencoder import Autoencoder, Decoder
from sklearn.metrics import accuracy_score
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, TensorDataset

type FeatureTargetEntry = tuple[torch.Tensor, torch.Tensor]


class MseKldLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self._mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return loss_MSE + loss_KLD


def encode_features(
    dataset_train: pd.DataFrame, dataset_test: pd.DataFrame
) -> tuple[torch.Tensor, torch.Tensor]:
    categorical_encoder = encoders.OneHotEncoder()
    numerical_encoder = encoders.MinMaxNumericalEncoder()

    dataset_train = preprocessing.encode_dataset(
        dataset_train,
        categorical_encoder=categorical_encoder.train_and_transform_feature,
        numerical_encoder=numerical_encoder.train_and_transform_feature,
    )

    dataset_test = preprocessing.encode_dataset(
        dataset_test,
        categorical_encoder=categorical_encoder.transform_feature,
        numerical_encoder=numerical_encoder.transform_feature,
    )

    return preprocessing.as_tensor(dataset_train), preprocessing.as_tensor(dataset_test)


def encode_target(
    dataset_train: pd.DataFrame, dataset_test: pd.DataFrame
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        preprocessing.as_tensor(dataset_train).long(),
        preprocessing.as_tensor(dataset_test).long(),
    )


def prepare_dataset(
    dataset: pd.DataFrame,
    train_test_split: float,
) -> tuple[Dataset[FeatureTargetEntry], Dataset[FeatureTargetEntry]]:
    client_datasets_features_train, client_datasets_features_test = (
        preprocessing.build_client_datasets(
            dataset,
            train_test_split,
        )
    )

    encoded_features_train, encoded_features_test = encode_features(
        client_datasets_features_train, client_datasets_features_test
    )

    dataset_target_train, dataset_target_test = preprocessing.build_target_dataset(
        dataset,
        train_test_split,
    )

    encoded_target_train, encoded_target_test = encode_target(
        dataset_target_train, dataset_target_test
    )

    return (
        TensorDataset(encoded_features_train, encoded_target_train),
        TensorDataset(encoded_features_test, encoded_target_test),
    )


type Minibatch = list[torch.Tensor]


def train_teacher(
    model: nn.Module,
    epochs: int,
    loader: DataLoader[FeatureTargetEntry],
    optimizer: torch.optim.Optimizer,
):
    loss_function = MseKldLoss()

    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()

        total_loss = 0.0
        batches = 0

        for batch_data in loader:
            batches += 1

            outs, mu, logvar = model.forward(batch_data)

            merged_minibatch_data = torch.concat(batch_data, dim=1)
            loss = loss_function(outs, merged_minibatch_data, mu, logvar)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch} Loss: {total_loss / batches:.3f}")


def train_student(
    student_model: Decoder,
    teacher_model: Autoencoder,
    epochs: int,
    epoch_size: int,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
):
    batches = epoch_size // batch_size
    loss_function = nn.MSELoss()

    student_model.train()
    teacher_model.eval()

    for epoch in range(epochs):
        total_loss = 0
        
        for _ in range(batches):
            latent = teacher_model.sample_latent(epoch_size).detach()
            optimizer.zero_grad()

            teacher_output = teacher_model.sample_from_latent(latent)
            student_output = student_model.sample_from_latent(latent)

            loss = loss_function(merge(student_output), merge(teacher_output))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch: {epoch} Loss: {total_loss / batches :.3f}")


def merge(tensors: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.concat(tensors, dim=1)


def _run_evaluator_model(
    dataset_train: Dataset[FeatureTargetEntry],
    dataset_test: Dataset[FeatureTargetEntry],
):
    features_train, targets_train = _load_dataset_as_tensor(dataset_train)
    features_test, targets_test = _load_dataset_as_tensor(dataset_test)

    model = EvaluatorModel()
    optimizer = optim.AdamW(model.parameters())

    loss_function = nn.CrossEntropyLoss()

    for epoch in range(1, 50):
        # train evaluator model
        optimizer.zero_grad()

        outputs_train = model(features_train)
        loss = loss_function(outputs_train, targets_train.squeeze(dim=1))

        loss.backward()
        optimizer.step()

        predictions_train = torch.argmax(outputs_train, 1)
        train_accuracy = accuracy_score(targets_train, predictions_train)

        # test evaluator model
        outputs_test = model(features_test)

        predictions_test = torch.argmax(outputs_test, 1)
        test_accuracy = accuracy_score(targets_test, predictions_test)

        print(
            "Epoch {}, Loss: {:.2f}, Acc:{:.2f}%, Test Acc: {:.2f}%".format(
                epoch, loss.item(), train_accuracy * 100, test_accuracy * 100
            )
        )


def _load_dataset_as_tensor(dataset: Dataset[FeatureTargetEntry]) -> FeatureTargetEntry:
    dataset_size = len(dataset)

    loader = DataLoader(dataset, batch_size=dataset_size)
    return next(iter(loader))


def _evaluate_decoder(
    *,
    model: Decoder | Autoencoder,
    dataset_train: Dataset[FeatureTargetEntry],
    dataset_test: Dataset[FeatureTargetEntry],
):
    model.eval()

    dataset_train_tensor = _load_dataset_as_tensor(dataset_train)

    latent = model.sample_latent(dataset_train_tensor[0].shape[0])
    synthetic_x, synthetic_y = model.sample_from_latent(latent)

    _run_evaluator_model(TensorDataset(synthetic_x.detach(), synthetic_y.detach()), dataset_test)


def main(
    *,
    train_test_split: float = 0.8,
):
    np.random.seed(42)
    torch.manual_seed(42)

    dataset = preprocessing.load_dataset(
        Path(__file__).parent.parent / "datasets" / "heart" / "dataset.csv"
    )

    dataset_train, dataset_test = prepare_dataset(dataset, train_test_split)

    dataset_train_complete = _load_dataset_as_tensor(dataset_train)
    dataset_train_size = dataset_train_complete[0].shape[0]
    dataset_train_features_count = dataset_train_complete[0].shape[1] + 1

    teacher_model = Autoencoder(
        input_dimensions=dataset_train_features_count,
        wide_hidden_dimensions=48,
        narrow_hidden_dimensions=32,
        latent_dimensions=16,
    )

    student_model = Decoder(
        output_dimensions=dataset_train_features_count,
        wide_hidden_dimensions=48,
        narrow_hidden_dimensions=32,
        latent_dimensions=16,
    )

    print("==============Baseline==============")
    _run_evaluator_model(dataset_train, dataset_test)

    print("==============Training teacher model==========")
    train_teacher(
        teacher_model,
        epochs=200,
        loader=DataLoader(dataset_train, batch_size=64),
        optimizer=optim.Adam(teacher_model.parameters(), lr=1e-3),
    )

    print("==============Evaluating teacher model==========")
    _evaluate_decoder(
        model=teacher_model,
        dataset_train=dataset_train,
        dataset_test=dataset_test,
    )

    print("==============Training student model==========")
    train_student(
        student_model=student_model,
        teacher_model=teacher_model,
        epochs=150,
        epoch_size=dataset_train_size,
        batch_size=128,
        optimizer=optim.Adam(student_model.parameters(), lr=1e-3),
    )

    print("==============Evaluating student model==========")
    _evaluate_decoder(
        model=student_model,
        dataset_train=dataset_train,
        dataset_test=dataset_test,
    )


if __name__ == "__main__":
    main()
