import typing

import torch
from torch import nn


class LossFunction(typing.Protocol):
    def __call__(
        self,
        outs: torch.Tensor,
        minibatch_data: torch.Tensor,
        mean: torch.Tensor,
        log_variance: torch.Tensor,
    ) -> torch.Tensor: ...


class Autoencoder(nn.Module):
    def __init__(
        self,
        input_dimensions: int,
        wide_hidden_dimensions: int = 32,
        narrow_hidden_dimensions: int = 12,
        latent_dimensions: int = 8,
    ):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Sequential(
                nn.Linear(input_dimensions, wide_hidden_dimensions),
                nn.BatchNorm1d(num_features=wide_hidden_dimensions),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Linear(wide_hidden_dimensions, narrow_hidden_dimensions),
                nn.BatchNorm1d(num_features=narrow_hidden_dimensions),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Linear(narrow_hidden_dimensions, narrow_hidden_dimensions),
                nn.BatchNorm1d(num_features=narrow_hidden_dimensions),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Linear(narrow_hidden_dimensions, latent_dimensions),
                nn.BatchNorm1d(num_features=latent_dimensions),
                nn.ReLU(),
            ),
        )

        self.latent_layer_mean = nn.Linear(latent_dimensions, latent_dimensions)
        self.latent_layer_log_variance = nn.Linear(latent_dimensions, latent_dimensions)

        self.decoder = nn.Sequential(
            nn.Sequential(
                nn.Linear(latent_dimensions, latent_dimensions),
                nn.BatchNorm1d(latent_dimensions),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Linear(latent_dimensions, narrow_hidden_dimensions),
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
                nn.Linear(wide_hidden_dimensions, input_dimensions),
                nn.BatchNorm1d(num_features=input_dimensions),
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
        std = log_variance.mul(0.5).exp_()
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def decode(self, reparameterized_latent_representation: torch.Tensor):
        return self.decoder(reparameterized_latent_representation)

    # TODO: Do transformation elsewhere; ideally in wrapper module
    def forward(
        self, x: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.concat(x, dim=1)

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @property
    def latent_dimensions(self) -> int:
        return self.latent_layer_mean.in_features

    def sample_latent(self, samples: int) -> torch.Tensor:
        sigma = torch.ones(self.latent_dimensions)
        mu = torch.zeros(self.latent_dimensions)

        distribution = torch.distributions.Normal(mu, sigma)
        return distribution.rsample(sample_shape=torch.Size([samples]))

    def sample_from_latent(
        self, latent: torch.Tensor, is_mnist: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # sigma = torch.exp(logvar / 2)
        output: torch.Tensor = self.decode(latent)

        # Predicted target probably does not have value one of the permitted values ('0' or '1')
        # We first clip the target value to ensure that it lies in [0, 1] and then round it to obtain either '0' or '1'
        # Rounding without clipping may yield other integer values
        if is_mnist:
            output[:, -1].clip_(0, 9)
            output[:, -1].round_()
        else:
            output[:, -1].clip_(0, 1)
            output[:, -1].round_()

        return (
            output[:, :-1],
            output[:, -1].unsqueeze(dim=1).long(),
        )


class Decoder(nn.Module):
    def __init__(
        self,
        output_dimensions: int,
        wide_hidden_dimensions: int = 32,
        narrow_hidden_dimensions: int = 12,
        latent_dimensions: int = 8,
    ):
        super(Decoder, self).__init__()
        self.latent_dimensions = latent_dimensions

        self.layer = nn.Sequential(
            nn.Sequential(
                nn.Linear(latent_dimensions, latent_dimensions),
                nn.BatchNorm1d(latent_dimensions),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Linear(latent_dimensions, narrow_hidden_dimensions),
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

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.layer(latent)

    def sample_latent(self, samples: int) -> torch.Tensor:
        sigma = torch.ones(self.latent_dimensions)
        mu = torch.zeros(self.latent_dimensions)

        distribution = torch.distributions.Normal(mu, sigma)
        return distribution.rsample(sample_shape=torch.Size([samples]))

    def sample_from_latent(
        self, latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output: torch.Tensor = self(latent)

        # Predicted target probably does not have value one of the permitted values ('0' or '1')
        # We first clip the target value to ensure that it lies in [0, 1] and then round it to obtain either '0' or '1'
        # Rounding without clipping may yield other integer values
        output[:, -1].clip_(0, 1)
        output[:, -1].round_()

        return (
            output[:, :-1],
            output[:, -1].unsqueeze(dim=1).long(),
        )


class MseKldLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._mse_loss = nn.MSELoss()

    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self._mse_loss(x_recon, x) + 2 * self._mse_loss(x_recon[:, -1], x[:, -1])
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return loss_MSE, loss_KLD
