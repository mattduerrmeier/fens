import typing

import torch

from .model import Autoencoder, Decoder
from .conditional_model import ConditionalAutoencoder, ConditionalDecoder


def sample_labels(samples: int, num_labels: int, device: torch.device) -> torch.Tensor:
    labels = torch.arange(num_labels).repeat(samples // num_labels + 1)[:samples]
    labels = labels[torch.randperm(samples)]  # shuffle for randomness
    labels = labels.to(device)
    labels = torch.nn.functional.one_hot(labels, num_classes=num_labels).float()

    return labels


def sample_proxy_dataset_tensor(
    models: list[None | typing.Union[ConditionalAutoencoder, ConditionalDecoder]],
    num_labels: int,
    samples: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_clients = len(models)
    num_labels_per_client = num_labels / num_clients

    latents: list[torch.Tensor] = []
    outputs: list[torch.Tensor] = []

    for client_idx, model in enumerate(models):
        if model is None:
            continue

        client_labels_start = int(client_idx * num_labels_per_client)
        client_labels_end = int((client_idx + 1) * num_labels_per_client)

        num_client_labels = client_labels_end - client_labels_start

        labels = torch.arange(client_labels_start, client_labels_end).repeat(
            samples // num_client_labels + 1
        )[:samples]
        labels = labels[torch.randperm(samples)]  # shuffle for randomness
        labels = labels.to(device)
        labels = torch.nn.functional.one_hot(labels, num_classes=num_labels).float()

        model.eval()
        latent = model.sample_latent(samples)
        latent = latent.to(device)
        synthetic_x = model.sample_from_latent(latent, labels)

        latents.append(latent)

        # TODO: HACK to keep required changes low
        outputs.append(torch.cat((synthetic_x.detach(), labels.detach()), dim=1))

    return torch.stack(latents), torch.stack(outputs)


def convert_tensor_to_dataset(
    downstream_dataset: torch.Tensor, num_labels: int
) -> torch.utils.data.TensorDataset:
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
