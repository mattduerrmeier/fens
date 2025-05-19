import torch
from autoencoder.model import Decoder
from train import evaluate_downstream_task
import matplotlib.pyplot as plt
import numpy as np

from .common import AggregatorResult, sample_proxy_dataset


def run_and_evaluate(
    model_config: dict[str, int],
    test_loader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    proxy_latents_tensor: torch.Tensor,
    proxy_dataset_tensor: torch.Tensor,
    num_labels: int,
    device: torch.device,
) -> AggregatorResult:
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
        model_config,
        proxy_dataset,
        epochs=50,
        batch_size=64,
        num_labels=num_labels,
        device=device,
    )

    # get loss
    _, downstream_proxy_data = sample_proxy_dataset([best_student_model], 1000, device)

    downstream_train_accuracy, downstream_test_accuracy = evaluate_on_downstream_task(
        downstream_proxy_data, test_loader, num_labels, device
    )

    visualize(downstream_proxy_data, num_labels)

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
    optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-3)

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

        print(f"Epoch: {epoch} Loss: {total_loss / batches:.3f}")

    return student_model


def convert_downstream_data_to_dataset(
    downstream_dataset: torch.Tensor, num_labels: int
) -> torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]:
    downstream_dataset = downstream_dataset.flatten(end_dim=1)

    synthetic_x: torch.Tensor
    synthetic_y: torch.Tensor
    if num_labels > 2:
        synthetic_x = downstream_dataset[:, :-num_labels]
        synthetic_y = (
            downstream_dataset[:, -num_labels:]
            .clip(0, 1)
            .round()
            .argmax(dim=1, keepdim=True)
        )
    else:
        synthetic_x = downstream_dataset[:, :-1]
        synthetic_y = downstream_dataset[:, -1].unsqueeze(dim=1).clip(0, 1).round()

    return torch.utils.data.TensorDataset(synthetic_x, synthetic_y)


def evaluate_on_downstream_task(
    downstream_dataset_tensor: torch.Tensor,
    test_loader: torch.utils.data.DataLoader,
    num_labels: int,
    device: torch.device,
) -> tuple[float, float]:
    downstream_dataset = convert_downstream_data_to_dataset(
        downstream_dataset_tensor, num_labels
    )

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


def render_image(image_tensor: torch.Tensor, label: torch.Tensor, axis) -> None:
    image_tensor = image_tensor.detach().cpu()

    axis.set_title(f"label: {label.item()}")
    axis.imshow(image_tensor.reshape(28, 28, 1), cmap="gray")


def visualize_from_dataset(
    downstream_dataset: torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]],
) -> None:
    fig, axes = plt.subplots(10, 10, figsize=(15, 15))

    loader = torch.utils.data.DataLoader(downstream_dataset, batch_size=1, shuffle=True)
    loader_iter = iter(loader)

    for (image_tensor, label), axis in zip(loader_iter, axes.flatten()):
        render_image(image_tensor, label, axis)

    plt.title("VAE Latent Space")
    plt.savefig("out/mnist-synthetic.png")


def visualize(
    dataset_tensor: torch.Tensor,
    num_labels: int,
) -> None:
    downstream_dataset = convert_downstream_data_to_dataset(dataset_tensor, num_labels)

    fig, axes = plt.subplots(10, 10, figsize=(15, 15))

    loader = torch.utils.data.DataLoader(downstream_dataset, batch_size=1, shuffle=True)
    loader_iter = iter(loader)

    for (image_tensor, label), axis in zip(loader_iter, axes.flatten()):
        render_image(image_tensor, label, axis)

    plt.title("VAE Latent Space")
    plt.savefig("out/mnist-synthetic.png")


def visualize_synthetic_data(
    dataset_tensor: torch.Tensor,
    num_labels: int,
    digit_size: int = 28,
    num_samples: int = 10,
    scale: float = 1.0,
    figsize: tuple[float, float] = (15, 15),
) -> None:
    downstream_dataset = convert_downstream_data_to_dataset(dataset_tensor, num_labels)

    figure = np.zeros((digit_size * num_samples, digit_size * num_samples))

    grid_x = np.linspace(-scale, scale, num_samples)
    grid_y = np.linspace(-scale, scale, num_samples)[::-1]

    loader = torch.utils.data.DataLoader(downstream_dataset, batch_size=1, shuffle=True)
    loader_iter = iter(loader)

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            image, label = next(loader_iter)

            digit = image.detach().cpu()

            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit[0][0]

    plt.figure(figsize=figsize)
    plt.title("VAE Latent Space")

    plt.xlabel("z [0]")
    plt.ylabel("z [1]")

    plt.imshow(figure, cmap="gray")

    plt.savefig("out/mnist-synthetic.png")
