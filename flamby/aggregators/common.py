import typing

import torch
from autoencoder.model import Autoencoder, Decoder


class AggregatorResult(typing.TypedDict):
    mse_loss: float
    downstream_train_accuracy: float
    downstream_test_accuracy: float


def sample_proxy_dataset(
    models: list[typing.Union[Autoencoder, Decoder]],
    samples: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    latents: list[torch.Tensor] = []
    outputs: list[torch.Tensor] = []

    for model in models:
        latent = model.sample_latent(samples)
        latent = latent.to(device)
        synthetic_x, synthetic_y = model.sample_from_latent(latent)

        latents.append(latent)
        outputs.append(
            torch.cat(
                (synthetic_x.detach(), synthetic_y.detach().clip(0, 1).round()), dim=1
            )
        )

    return torch.stack(latents), torch.stack(outputs)
