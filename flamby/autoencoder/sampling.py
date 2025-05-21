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

        prev_idx = 0
        synthetic_x, synthetic_y = [], []

        for next_idx in range(samples//10, samples+1, samples//10):
            t = latent[prev_idx:next_idx].to(device)
            prev_idx = next_idx
            synth_x, synth_y = model.sample_from_latent(t, requires_argmax=False)
            synthetic_x.append(synth_x.detach().cpu())
            synthetic_y.append(synth_y.detach().cpu())

        synthetic_x = torch.cat(synthetic_x)
        synthetic_y = torch.cat(synthetic_y)

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
