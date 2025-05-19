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


class StolenAutoencoder(torch.nn.Module):
    def __init__(
        self,
        input_dimensions: int,
        num_classes: int,
        wide_hidden_dimensions=400,
        narrow_hidden_dimensions=200,
        latent_dimensions: int = 999,
    ):
        super().__init__()

        self.num_classes = num_classes

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dimensions, wide_hidden_dimensions),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(wide_hidden_dimensions, narrow_hidden_dimensions),
            torch.nn.ReLU(inplace=True),
        )

        self.to_mean = torch.nn.Linear(narrow_hidden_dimensions, latent_dimensions)
        self.to_log_variance = torch.nn.Linear(
            narrow_hidden_dimensions, latent_dimensions
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(mean_logvar_dim, narrow_hidden_dimensions),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(narrow_hidden_dimensions, wide_hidden_dimensions),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(wide_hidden_dimensions, input_dimensions),
            torch.nn.Sigmoid(),
        )

    def forward(self, tensor):
        tensor = torch.concat(tensor, dim=1)

        z_mean, z_log_variance = self.encode(tensor)

        z = self.sample_normal(z_mean, z_log_variance)

        out_tensor = self.decode(z)

        return out_tensor, z_mean, z_log_variance

    def encode(self, tensor):
        tensor = self.encoder(tensor)
        mean, log_variance = self.to_mean(tensor), self.to_log_variance(tensor)

        return mean, log_variance

    def decode(self, tensor):
        tensor = self.decoder(tensor)

        return tensor

    def sample_normal(self, mean, log_variance):
        std = torch.exp(0.5 * log_variance)
        epsilon = torch.randn_like(std)

        return mean + epsilon * std

    def sample_latent(self, samples: int) -> torch.Tensor:
        sigma = torch.ones(2)
        mu = torch.zeros(2)

        distribution = torch.distributions.Normal(mu, sigma)
        return distribution.rsample(sample_shape=torch.Size([samples]))

    def sample_from_latent(
        self, latent: torch.Tensor, requires_argmax: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # sigma = torch.exp(logvar / 2)
        output: torch.Tensor = self.decode(latent)

        # Predicted target probably does not have value one of the permitted values ('0' or '1')
        # We first clip the target value to ensure that it lies in [0, 1] and then round it to obtain either '0' or '1'
        # Rounding without clipping may yield other integer values
        synthetic_x: torch.Tensor
        synthetic_y: torch.Tensor
        if self.num_classes > 2:
            # output[:, -self.num_classes :].clip_(0, 1)
            # output[:, -self.num_classes :].round_()
            synthetic_x = output[:, : -self.num_classes]
            synthetic_y = output[:, -self.num_classes :]
            if requires_argmax:
                synthetic_y = synthetic_y.argmax(dim=1).unsqueeze(dim=1).long()
        else:
            output[:, -1].clip_(0, 1)
            output[:, -1].round_()
            synthetic_x = output[:, :-1]
            synthetic_y = output[:, -1].unsqueeze(dim=1).long()

        return synthetic_x, synthetic_y


class Autoencoder(nn.Module):
    def __init__(
        self,
        input_dimensions: int,
        num_classes: int,
        wide_hidden_dimensions: int = 400,
        narrow_hidden_dimensions: int = 200,
        latent_dimensions: int = 2,
    ):
        super(Autoencoder, self).__init__()
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
                nn.Linear(latent_dimensions, narrow_hidden_dimensions),
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

    def sample_latent(self, samples: int) -> torch.Tensor:
        sigma = torch.ones(self.latent_dimensions)
        mu = torch.zeros(self.latent_dimensions)

        distribution = torch.distributions.Normal(mu, sigma)
        return distribution.rsample(sample_shape=torch.Size([samples]))

    def sample_from_latent(
        self, latent: torch.Tensor, requires_argmax: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # sigma = torch.exp(logvar / 2)
        output: torch.Tensor = self.decode(latent)

        # Predicted target probably does not have value one of the permitted values ('0' or '1')
        # We first clip the target value to ensure that it lies in [0, 1] and then round it to obtain either '0' or '1'
        # Rounding without clipping may yield other integer values
        synthetic_x: torch.Tensor
        synthetic_y: torch.Tensor
        if self.num_classes > 2:
            # output[:, -self.num_classes :].clip_(0, 1)
            # output[:, -self.num_classes :].round_()
            synthetic_x = output[:, : -self.num_classes]
            synthetic_y = output[:, -self.num_classes :]
            if requires_argmax:
                synthetic_y = synthetic_y.argmax(dim=1).unsqueeze(dim=1).long()
        else:
            output[:, -1].clip_(0, 1)
            output[:, -1].round_()
            synthetic_x = output[:, :-1]
            synthetic_y = output[:, -1].unsqueeze(dim=1).long()

        return synthetic_x, synthetic_y


class Decoder(nn.Module):
    def __init__(
        self,
        output_dimensions: int,
        num_classes: int,
        wide_hidden_dimensions: int = 32,
        narrow_hidden_dimensions: int = 12,
        latent_dimensions: int = 8,
    ):
        super(Decoder, self).__init__()
        self.latent_dimensions = latent_dimensions
        self.num_classes = num_classes

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
        self,
        latent: torch.Tensor,
        requires_argmax: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output: torch.Tensor = self(latent)

        # Predicted target probably does not have value one of the permitted values ('0' or '1')
        # We first clip the target value to ensure that it lies in [0, 1] and then round it to obtain either '0' or '1'
        # Rounding without clipping may yield other integer values
        synthetic_x: torch.Tensor
        synthetic_y: torch.Tensor
        if self.num_classes > 2:
            # not clipping and rounding seems to help with performance
            # output[:, -self.num_classes :].clip_(0, 1)
            # output[:, -self.num_classes :].round_()
            synthetic_x = output[:, : -self.num_classes]
            synthetic_y = output[:, -self.num_classes :]
            if requires_argmax:
                synthetic_y = synthetic_y.argmax(dim=1).unsqueeze(dim=1).long()
        else:
            output[:, -1].clip_(0, 1)
            output[:, -1].round_()
            synthetic_x = output[:, :-1]
            synthetic_y = output[:, -1].unsqueeze(dim=1).long()

        return synthetic_x, synthetic_y


class MseKldLoss(nn.Module):
    def __init__(self, num_classes, target_coeff=2):
        super().__init__()
        self._mse_loss = nn.MSELoss(reduction="sum")
        self.num_classes = num_classes if num_classes > 2 else 1
        self.target_coeff = target_coeff

    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self._mse_loss(x_recon, x) + self.target_coeff * self._mse_loss(
            x_recon[:, -self.num_classes :], x[:, -self.num_classes :]
        )

        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return loss_MSE, loss_KLD
