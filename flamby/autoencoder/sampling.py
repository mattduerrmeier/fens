import typing

import torch

from .model import Autoencoder, Decoder


def sample_proxy_dataset_tensor(
    models: list[typing.Union[Autoencoder, Decoder]],
    samples: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    latents: list[torch.Tensor] = []
    outputs: list[torch.Tensor] = []

    for model in models:
        model.eval()
        latent = model.sample_latent(samples)
        latent = latent.to(device)
        synthetic_x, synthetic_y = model.sample_from_latent(
            latent, requires_argmax=False
        )

        latents.append(latent)
        outputs.append(torch.cat((synthetic_x.detach(), synthetic_y.detach()), dim=1))

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
