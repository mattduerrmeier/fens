import torch
import torch.nn.functional as F
from torch import nn


class VAE(nn.Module):
    """
    Variational Autoencder class.
    Each clients train a VAE locally.
    """

    def __init__(self, D_in, H=48, H2=32, latent_dim=16):
        super().__init__()
        # encoder part
        self.linear1 = nn.Linear(D_in, H)
        self.lin_bn1 = nn.BatchNorm1d(num_features=H)
        self.linear2 = nn.Linear(H, H2)
        self.lin_bn2 = nn.BatchNorm1d(num_features=H2)
        self.linear3 = nn.Linear(H2, H2)
        self.lin_bn3 = nn.BatchNorm1d(num_features=H2)

        # latent vectors mu and sigma
        self.fc1 = nn.Linear(H2, latent_dim)
        self.bn1 = nn.BatchNorm1d(num_features=latent_dim)
        self.fc21 = nn.Linear(latent_dim, latent_dim)
        self.fc22 = nn.Linear(latent_dim, latent_dim)

        # sampling vector
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc_bn3 = nn.BatchNorm1d(latent_dim)
        self.fc4 = nn.Linear(latent_dim, H2)
        self.fc_bn4 = nn.BatchNorm1d(H2)

        # decoder part
        self.linear4 = nn.Linear(H2, H2)
        self.lin_bn4 = nn.BatchNorm1d(num_features=H2)
        self.linear5 = nn.Linear(H2, H)
        self.lin_bn5 = nn.BatchNorm1d(num_features=H)
        self.linear6 = nn.Linear(H, D_in)
        self.lin_bn6 = nn.BatchNorm1d(num_features=D_in)

    def encode(self, x):
        x = F.relu(self.lin_bn1(self.linear1(x)))
        x = F.relu(self.lin_bn2(self.linear2(x)))
        x = F.relu(self.lin_bn3(self.linear3(x)))

        out = F.relu(self.bn1(self.fc1(x)))
        out1 = self.fc21(out)
        out2 = self.fc22(out)
        return (out1, out2)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, x):
        x = F.relu(self.fc_bn3(self.fc3(x)))
        x = F.relu(self.fc_bn4(self.fc4(x)))

        x = F.relu(self.lin_bn4(self.linear4(x)))
        x = F.relu(self.lin_bn5(self.linear5(x)))
        out = self.lin_bn6(self.linear6(x))
        return out

    def forward(self, x):
        x = torch.concat(x, dim=1)

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
        return out, mu, logvar


class MseKldLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self._mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return loss_MSE + loss_KLD
