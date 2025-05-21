import logging
import typing

import torch
from autoencoder import visualization
from autoencoder.model import Decoder
from autoencoder.sampling import convert_tensor_to_dataset, sample_proxy_dataset_tensor
from params.visualization import VisualizationParameters
from train import evaluate_downstream_task
from utils import determine_label_distribution

from .common import AggregatorResult


def run_and_evaluate(
    agg_params: typing.Mapping[str, typing.Any],
    test_loader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    proxy_latents_tensor: torch.Tensor,
    proxy_dataset_tensor: torch.Tensor,
    num_labels: int,
    visualization_parameters: VisualizationParameters,
    device: torch.device,
) -> AggregatorResult:
    logging.info("Teaching student model")

    if num_labels > 2:
        x_proxy = proxy_dataset_tensor.flatten(end_dim=1)[:, :-num_labels]
        y_proxy = proxy_dataset_tensor.flatten(end_dim=1)[:, -num_labels:]
    else:
        x_proxy = proxy_dataset_tensor.flatten(end_dim=1)[:, :-1]
        y_proxy = proxy_dataset_tensor.flatten(end_dim=1)[:, -1:]

    proxy_dataset = torch.utils.data.TensorDataset(
        proxy_latents_tensor.flatten(end_dim=1),
        x_proxy,
        y_proxy,
    )

    best_student_model = train_student(
        agg_params["model_config"],
        proxy_dataset,
        epochs=agg_params["distillation_epochs"],
        lr=agg_params["distillation_lr"],
        batch_size=64,
        num_labels=num_labels,
        device=device,
    )

    _, downstream_task_dataset_tensor = sample_proxy_dataset_tensor(
        [best_student_model], 10_000, device
    )
    downstream_task_dataset = convert_tensor_to_dataset(
        downstream_task_dataset_tensor, num_labels
    )

    logging.info(
        f"Generated {len(downstream_task_dataset)} samples from student model "
        "for training of downstream task "
        f"with label distribution {determine_label_distribution(downstream_task_dataset)} "
    )

    if visualization_parameters.supports_visualization:
        logging.info("Visualizing samples from student model")
        visualization.visualize_from_dataset(
            visualization_parameters.results_path / "aggregator-samples.png",
            downstream_task_dataset,
        )

        logging.info("Visualizing latent space of student model")
        visualization.visualize_latent(
            visualization_parameters.results_path / "aggregator-latent.png",
            best_student_model,
            device,
        )
    else:
        logging.info("Skip visualization of student model as dataset is not visual")

    logging.info("Evaluating student model on downstream task")
    downstream_train_accuracy, downstream_test_accuracy = evaluate_on_downstream_task(
        downstream_task_dataset, test_loader, num_labels, device
    )

    return {
        "mse_loss": -1,
        "downstream_train_accuracy": downstream_train_accuracy,
        "downstream_test_accuracy": downstream_test_accuracy,
    }


def train_student(
    model_config: dict[str, int],
    proxy_dataset: torch.utils.data.Dataset[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ],
    epochs: int,
    lr: float,
    batch_size: int,
    num_labels: int,
    device: torch.device,
) -> torch.nn.Module:
    # output_dim = input size + number of classes
    output_dimensions = len(proxy_dataset[0][1]) + len(proxy_dataset[0][2])
    student_model = Decoder(
        output_dimensions=output_dimensions,
        num_classes=num_labels,
        **model_config,
    ).to(device)
    optimizer = torch.optim.Adam(student_model.parameters(), lr=lr)

    batches = len(proxy_dataset) // batch_size

    mse_loss = torch.nn.MSELoss(reduction="sum")

    def adapted_loss(
        actual_output: tuple[torch.Tensor, torch.Tensor],
        expected_output: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        x_loss = mse_loss(actual_output[0], expected_output[0])
        y_loss = mse_loss(actual_output[1], expected_output[1].float())
        return x_loss + y_loss

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
            student_output = student_model.sample_from_latent(
                batch_latents, requires_argmax=False
            )

            loss = loss_function(student_output, teacher_output)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        logging.info(f"Epoch: {epoch} Loss: {total_loss / batches:.3f}")

    return student_model


def evaluate_on_downstream_task(
    downstream_dataset: torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]],
    test_loader: torch.utils.data.DataLoader,
    num_labels: int,
    device: torch.device,
) -> tuple[float, float]:
    train_loader = torch.utils.data.DataLoader(
        downstream_dataset,
        batch_size=32,
    )

    train_accuracy, test_accuracy = evaluate_downstream_task(
        "distillation",
        train_loader,
        test_loader,
        num_labels,
        device,
    )

    return train_accuracy, test_accuracy
