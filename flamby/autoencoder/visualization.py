import pathlib

import numpy as np
import torch
from matplotlib import pyplot as plt

from .model import Decoder
from .conditional_model import ConditionalDecoder
from .sampling import sample_labels


def render_image(image_tensor: torch.Tensor, label: torch.Tensor, axis) -> None:
    image_tensor = image_tensor.detach().cpu()

    # digit = label.argmax()
    # axis.set_title(f"label: {digit}")
    axis.axis("off")
    axis.imshow(image_tensor.reshape(28, 28, 1), cmap="gray")


def visualize_from_dataset(
    path: pathlib.Path,
    downstream_dataset: torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]],
) -> None:
    fig, axes = plt.subplots(10, 10, figsize=(15, 15))
    fig.suptitle("VAE: randomly generated samples")

    loader = torch.utils.data.DataLoader(downstream_dataset, batch_size=1, shuffle=True)
    loader_iter = iter(loader)

    for (image_tensor, label), axis in zip(loader_iter, axes.flatten()):
        render_image(image_tensor, label, axis)

    fig.savefig(path)


def visualize_latent_for_digit(
    fig, digit: int, model: ConditionalDecoder, device: torch.device
) -> None:
    model.eval()

    axes = fig.subplots(6, 6)
    fig.suptitle(f"latent space for digit {digit}")

    grid_x = np.linspace(-4, 4, 6, dtype=np.float32)
    grid_y = np.linspace(-4, 4, 6, dtype=np.float32)[::-1]

    for row_idx, latent_value_1 in enumerate(grid_x):
        for column_idx, latent_value_2 in enumerate(grid_y):
            cell_latent = (
                torch.tensor([latent_value_1, latent_value_2])
                .float()
                .unsqueeze(dim=0)
                .to(device)
            )

            # TODO: Hard-coded number of classes
            cell_label = torch.nn.functional.one_hot(
                torch.tensor([digit], device=device), num_classes=10
            )
            cell_image = model.sample_from_latent(cell_latent, cell_label)

            axis = axes[row_idx, column_idx]
            render_image(cell_image, cell_label, axis)

    latent_value_axis = fig.add_subplot()
    latent_value_axis.set_xticks(np.arange(-4, 4, 1))
    latent_value_axis.set_yticks(np.arange(-4, 4, 1))
    latent_value_axis.set_xlabel("1st dimension of latent")
    latent_value_axis.set_ylabel("2nd dimension of latent")

    latent_value_axis.set_zorder(-1)


def visualize_latent(
    path: pathlib.Path, model: ConditionalDecoder, device: torch.device
):
    fig = plt.figure(layout="constrained", figsize=(40, 30))
    subfigs = fig.subfigures(4, 3, wspace=0.07)

    for digit, subfig in enumerate(subfigs.flatten()):
        if digit > 9:
            continue

        visualize_latent_for_digit(subfig, digit, model, device)

    fig.savefig(path)
    plt.close(fig)
