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
        model.eval()
        latent = model.sample_latent(samples)
        
        latent = latent.to(device)
        synthetic_x, synthetic_y = model.sample_from_latent(
            latent, requires_argmax=False
        )

        latents.append(latent)
        outputs.append(torch.cat((synthetic_x.detach(), synthetic_y.detach()), dim=1))

    return torch.stack(latents), torch.stack(outputs)
