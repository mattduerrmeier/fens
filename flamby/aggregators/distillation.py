import torch
import typing
from .common import sample_proxy_dataset, AggregatorResult
from autoencoder.model import Autoencoder, Decoder
from train import evaluate_downstream_task


def run_and_evaluate(
    test_loader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    proxy_latents_tensor: torch.Tensor,
    proxy_dataset_tensor: torch.Tensor,
    device: torch.device,
) -> AggregatorResult:
    best_student_model = train_student(
        proxy_dataset=torch.utils.data.TensorDataset(
            proxy_latents_tensor.flatten(end_dim=1),
            proxy_dataset_tensor.flatten(end_dim=1)[:, :-1],
            proxy_dataset_tensor.flatten(end_dim=1)[:, -1:],
        ),
        epochs=200,
        batch_size=64,
        device=device,
    )

    # get loss
    _, downstream_proxy_data = sample_proxy_dataset([best_student_model], 1000)

    downstream_train_accuracy, downstream_test_accuracy = evaluate_on_downstream_task(
        downstream_proxy_data, test_loader
    )

    return {
        "mse_loss": -1,
        "downstream_train_accuracy": downstream_train_accuracy,
        "downstream_test_accuracy": downstream_test_accuracy,
    }


def train_student(
    proxy_dataset: torch.utils.data.Dataset[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ],
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> torch.nn.Module:
    student_model = Decoder(output_dimensions=13 + 1).to(device)
    optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-3)

    batches = len(proxy_dataset) // batch_size

    mse_loss = torch.nn.MSELoss()

    def adapted_loss(
        actual_output: tuple[torch.Tensor, torch.Tensor],
        expected_output: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        return mse_loss(actual_output[0], expected_output[0]) + 2 * mse_loss(
            actual_output[1], expected_output[1].float()
        )

    loss_function = adapted_loss

    student_model.train()

    for epoch in range(epochs):
        total_loss = 0

        for batch_latents, batch_features, batch_targets in torch.utils.data.DataLoader(
            proxy_dataset, batch_size=batch_size
        ):
            batch_latents = batch_latents.to(device)
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()

            teacher_output = (batch_features, batch_targets)
            student_output = student_model.sample_from_latent(batch_latents)

            loss = loss_function(student_output, teacher_output)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch: {epoch} Loss: {total_loss / batches :.3f}")

    return student_model


def evaluate_on_downstream_task(
    downstream_dataset: torch.Tensor,
    test_loader: torch.utils.data.DataLoader,
) -> tuple[float, float]:
    downstream_dataset = downstream_dataset.flatten(end_dim=1)

    train_accuracy, test_accuracy = evaluate_downstream_task(
        "distillation",
        torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                downstream_dataset[:, :-1],
                downstream_dataset[:, -1].unsqueeze(dim=1).clip(0, 1).round(),
            ),
            batch_size=32,
        ),
        test_loader,
    )

    return train_accuracy, test_accuracy
