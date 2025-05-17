import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from flamby.autoencoder import MseKldLoss, Autoencoder, Decoder

type FeatureTargetEntry = tuple[torch.Tensor, torch.Tensor]


from flamby.datasets.fed_heart_disease import (
    FedHeartDisease as FedHeartDataset,
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

        print(f"Epoch: {epoch} Loss: {total_loss / batches:.3f}")


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
