import pathlib

import numpy as np
import torch
from matplotlib import pyplot as plt

from .model import Decoder


def render_image(image_tensor: torch.Tensor, label: torch.Tensor, axis) -> None:
    image_tensor = image_tensor.detach().cpu()

    axis.set_title(f"label: {label.item()}")
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


def visualize_latent(path: pathlib.Path, model: Decoder, device: torch.device):
    model.eval()

    fig, axes = plt.subplots(11, 11, figsize=(15, 15))
    fig.suptitle("VAE: latent space")

    grid_x = np.linspace(-1, 1, 11, dtype=np.float32)
    grid_y = np.linspace(-1, 1, 11, dtype=np.float32)[::-1]

    for row_idx, latent_value_1 in enumerate(grid_x):
        for column_idx, latent_value_2 in enumerate(grid_y):
            cell_latent = (
                torch.tensor([latent_value_1, latent_value_2])
                .float()
                .unsqueeze(dim=0)
                .to(device)
            )
            cell_image, cell_label = model.sample_from_latent(cell_latent)

            axis = axes[row_idx, column_idx]
            render_image(cell_image, cell_label, axis)

    latent_value_axis = fig.add_subplot()
    latent_value_axis.set_xticks(np.arange(-1, 1, 0.2))
    latent_value_axis.set_yticks(np.arange(-1, 1, 0.2))
    latent_value_axis.set_xlabel("1st dimension of latent")
    latent_value_axis.set_ylabel("2nd dimension of latent")

    latent_value_axis.set_zorder(-1)

    fig.savefig(path)
