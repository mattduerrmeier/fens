import typing

import numpy as np
import torch
from torch import nn



class ConditionalAutoencoder(nn.Module):
    def __init__(
        self,
        input_dimensions: int,
        num_classes: int,
        wide_hidden_dimensions: int,
        narrow_hidden_dimensions: int,
        latent_dimensions: int,
    ):
        super(ConditionalAutoencoder, self).__init__()
        self.num_classes = num_classes

        self.latent_dimensions = latent_dimensions

        self.encoder = nn.Sequential(
            nn.Sequential(
                nn.Linear(input_dimensions, wide_hidden_dimensions),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Linear(wide_hidden_dimensions, narrow_hidden_dimensions),
                nn.ReLU(),
            ),
        )

        self.latent_layer_mean = nn.Linear(narrow_hidden_dimensions, latent_dimensions)
        self.latent_layer_log_variance = nn.Linear(
            narrow_hidden_dimensions, latent_dimensions
        )

        self.decoder = nn.Sequential(
            nn.Sequential(
                nn.Linear(latent_dimensions + num_classes, narrow_hidden_dimensions),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Linear(narrow_hidden_dimensions, wide_hidden_dimensions),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Linear(wide_hidden_dimensions, input_dimensions),
                nn.Sigmoid(),
            ),
        )

    def encode(self, x):
        encoded = self.encoder(x)

        mean = self.latent_layer_mean(encoded)
        log_variance = self.latent_layer_log_variance(encoded)

        return mean, log_variance

    def reparameterize(self, mean, log_variance):
        if not self.training:
            # leave mean unchanged during inference
            return mean

        # create new samples based on the parameters predicted by the encoder
        std = log_variance.mul(0.5).exp()
        eps = torch.randn_like(std)
        return eps.mul(std).add(mean)

    def decode(
        self, reparameterized_latent_representation: torch.Tensor, label: torch.Tensor
    ):
        latent_with_label = torch.cat(
            (reparameterized_latent_representation, label), dim=1
        )

        return self.decoder(latent_with_label)

    # TODO: Do transformation elsewhere; ideally in wrapper module
    def forward(
        self,
        x: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features, target = x

        mu, logvar = self.encode(features)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, target), mu, logvar

    def sample_latent(self, samples: int) -> torch.Tensor:
        sigma = torch.ones(self.latent_dimensions)
        mu = torch.zeros(self.latent_dimensions)

        distribution = torch.distributions.Normal(mu, sigma)
        return distribution.rsample(sample_shape=torch.Size([samples]))

    def sample_from_latent(
        self, latent: torch.Tensor, label: torch.Tensor
    ) -> torch.Tensor:
        # sigma = torch.exp(logvar / 2)
        output: torch.Tensor = self.decode(latent, label)

        return output


class ConditionalDecoder(nn.Module):
    def __init__(
        self,
        output_dimensions: int,
        num_classes: int,
        wide_hidden_dimensions: int,
        narrow_hidden_dimensions: int,
        latent_dimensions: int,
    ):
        super(ConditionalDecoder, self).__init__()
        self.latent_dimensions = latent_dimensions
        self.num_classes = num_classes

        self.layer = nn.Sequential(
            nn.Sequential(
                nn.Linear(latent_dimensions + num_classes, latent_dimensions + num_classes),
                nn.BatchNorm1d(latent_dimensions + num_classes),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Linear(latent_dimensions + num_classes, narrow_hidden_dimensions),
                nn.BatchNorm1d(narrow_hidden_dimensions),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Linear(narrow_hidden_dimensions, narrow_hidden_dimensions),
                nn.BatchNorm1d(num_features=narrow_hidden_dimensions),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Linear(narrow_hidden_dimensions, wide_hidden_dimensions),
                nn.BatchNorm1d(num_features=wide_hidden_dimensions),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Linear(wide_hidden_dimensions, output_dimensions),
                nn.BatchNorm1d(num_features=output_dimensions),
            ),
        )

    def forward(self, latent: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        latent_with_label = torch.cat((latent, label), dim=1)

        return self.layer(latent_with_label)

    def sample_latent(self, samples: int) -> torch.Tensor:
        sigma = torch.ones(self.latent_dimensions)
        mu = torch.zeros(self.latent_dimensions)

        distribution = torch.distributions.Normal(mu, sigma)
        return distribution.rsample(sample_shape=torch.Size([samples]))

    def sample_from_latent(
        self, latent: torch.Tensor, label: torch.Tensor
    ) -> torch.Tensor:
        # sigma = torch.exp(logvar / 2)
        output: torch.Tensor = self(latent, label)

        return output


class ConditionalMseKldLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self._mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return loss_MSE, loss_KLD
