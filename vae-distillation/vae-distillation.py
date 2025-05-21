import numpy as np
import pandas as pd
import torch
from components import encoders, preprocessing
from models.autoencoder import Autoencoder, Decoder
from sklearn.metrics import accuracy_score
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, TensorDataset

type FeatureTargetEntry = tuple[torch.Tensor, torch.Tensor]

from tqdm import tqdm


from flamby.datasets.fed_heart_disease import (
    FedHeartDisease as FedHeartDataset,
)


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

    for epoch in tqdm(range(epochs)):
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

    mse_loss = nn.MSELoss()

    def adapted_loss(
        actual_output: tuple[torch.Tensor, torch.Tensor],
        expected_output: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        return mse_loss(actual_output[0], expected_output[0]) + 2 * mse_loss(
            actual_output[1], expected_output[1].float()
        )

    loss_function = adapted_loss

    student_model.train()
    teacher_model.eval()

    for epoch in range(epochs):
        total_loss = 0

        for _ in range(batches):
            latent = teacher_model.sample_latent(epoch_size).detach()
            optimizer.zero_grad()

            teacher_output = teacher_model.sample_from_latent(latent)
            student_output = student_model.sample_from_latent(latent)

            loss = loss_function(student_output, teacher_output)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch: {epoch} Loss: {total_loss / batches :.3f}")


def merge(tensors: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.concat(tensors, dim=1)


class EvaluatorModel(nn.Module):
    def __init__(self, input_dimensions: int):
        super(EvaluatorModel, self).__init__()
        self.fc1 = nn.Linear(input_dimensions, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 2)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.dropout(x)

        return self.fc4(x)


def _run_evaluator_model(
    dataset_train: Dataset[FeatureTargetEntry],
    dataset_test: Dataset[FeatureTargetEntry],
    *,
    input_dimensions: int,
):
    features_train, targets_train = _load_dataset_as_tensor(dataset_train)
    features_test, targets_test = _load_dataset_as_tensor(dataset_test)

    model = EvaluatorModel(input_dimensions)
    optimizer = optim.AdamW(model.parameters())

    loss_function = nn.CrossEntropyLoss()

    for epoch in range(1, 50):
        # train evaluator model
        model.train()
        optimizer.zero_grad()

        outputs_train = model(features_train)
        loss = loss_function(outputs_train, targets_train.squeeze(dim=1).long())

        loss.backward()
        optimizer.step()

        predictions_train = torch.argmax(outputs_train, 1)
        train_accuracy = accuracy_score(targets_train, predictions_train)

        # test evaluator model
        model.eval()
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
    training_set_size: int,
    dataset_test: Dataset[FeatureTargetEntry],
    input_dimensions: int,
):
    model.eval()

    latent = model.sample_latent(training_set_size)
    synthetic_x, synthetic_y = model.sample_from_latent(latent)

    _run_evaluator_model(
        TensorDataset(synthetic_x.detach(), synthetic_y.detach()),
        dataset_test,
        input_dimensions=input_dimensions,
    )


def main():
    dataset = FedHeartDataset(normalize=True)

    encoded_dataset = TensorDataset(*list(torch.tensor(it) for it in dataset.features))

    normalized_dataset = FedHeartDataset(normalize=True)
    print(len(normalized_dataset.features))
    print(normalized_dataset.features[0])

    for i in range(5):
        entry_features, entry_target = dataset[i]
        print(f"#{i}", entry_features)

    for i in range(5):
        entry_features = encoded_dataset[i]
        print(f"#{i}", entry_features)

    for i in range(5):
        entry_features, entry_target = normalized_dataset[i]
        print(f"#{i}", entry_features)


def _main():
    dataset_train = FedHeartDataset(train=True)
    dataset_test = FedHeartDataset(train=False)

    for i in range(5):
        entry_features, entry_target = dataset_train[i]
        print(f"#{i}", entry_features)

    for i in range(5):
        entry_features, entry_target = dataset_test[i]
        print(f"#{i}", entry_features)


def main(
    *,
    train_test_split: float = 0.8,
):
    np.random.seed(42)
    torch.manual_seed(42)

    # dataset_train = FedHeartDataset(train=True)
    # dataset_test = FedHeartDataset(train=False)

    dataset = FedHeartDataset()
    dataset_train, dataset_test = torch.utils.data.random_split(dataset, (0.8, 0.2))

    dataset_train_complete = _load_dataset_as_tensor(dataset_train)
    dataset_train_size = dataset_train_complete[0].shape[0]
    dataset_train_features_count = dataset_train_complete[0].shape[1]

    teacher_model = Autoencoder(
        input_dimensions=dataset_train_features_count + 1,
        wide_hidden_dimensions=32,
        narrow_hidden_dimensions=12,
        latent_dimensions=8,
    )

    student_model = Decoder(
        output_dimensions=dataset_train_features_count + 1,
        wide_hidden_dimensions=32,
        narrow_hidden_dimensions=12,
        latent_dimensions=8,
    )

    print("==============Baseline==============")
    _run_evaluator_model(
        dataset_train, dataset_test, input_dimensions=dataset_train_features_count
    )

    print("==============Training teacher model==========")
    train_teacher(
        teacher_model,
        epochs=200,
        loader=DataLoader(dataset_train, batch_size=32),
        optimizer=optim.Adam(teacher_model.parameters(), lr=1e-3),
    )

    print("==============Evaluating teacher model==========")
    _evaluate_decoder(
        model=teacher_model,
        training_set_size=len(dataset_train),
        dataset_test=dataset_test,
        input_dimensions=dataset_train_features_count,
    )

    print("==============Training student model==========")
    train_student(
        student_model=student_model,
        teacher_model=teacher_model,
        epochs=200,
        epoch_size=dataset_train_size,
        batch_size=64,
        optimizer=optim.Adam(student_model.parameters(), lr=1e-3),
    )

    print("==============Evaluating student model==========")
    _evaluate_decoder(
        model=student_model,
        training_set_size=len(dataset_train),
        dataset_test=dataset_test,
        input_dimensions=dataset_train_features_count,
    )


if __name__ == "__main__":
    main()
