# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Shared components for VAE implementations
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np


# =============================================================================
# Priors
# =============================================================================

class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

        Parameters:
        M: [int] 
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class MixtureOfGaussiansPrior(nn.Module):
    def __init__(self, M, K=10):
        """
        Define a Mixture of Gaussians prior distribution.

        Parameters:
        M: [int] 
           Dimension of the latent space.
        K: [int]
           Number of components in the mixture.
        """
        super(MixtureOfGaussiansPrior, self).__init__()
        self.M = M
        self.K = K

        # Initialize mixture component means (learnable)
        self.means = nn.Parameter(torch.randn(K, M) * 0.5, requires_grad=True)

        # Initialize mixture component log-stds (learnable)
        self.log_stds = nn.Parameter(torch.zeros(K, M), requires_grad=True)

        # Initialize mixture weights (logits, learnable)
        self.mixture_logits = nn.Parameter(torch.zeros(K), requires_grad=True)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        mix = td.Categorical(logits=self.mixture_logits)
        comp = td.Independent(td.Normal(loc=self.means, scale=torch.exp(self.log_stds)), 1)
        return td.MixtureSameFamily(mix, comp)


# =============================================================================
# Encoder
# =============================================================================

class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]             
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


# =============================================================================
# Decoders
# =============================================================================

class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters: 
        decoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M)` as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.std = nn.Parameter(torch.ones(28, 28) * 0.5, requires_grad=True)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)


class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net, learn_variance=True, fixed_variance=0.1):
        """
        Define a Gaussian decoder distribution based on a given decoder network.

        Parameters: 
        decoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M)` as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2) or
           (batch_size, 2*feature_dim1, feature_dim2) if learning variance.
        learn_variance: [bool]
           If True, the network outputs both mean and log-variance.
           If False, use a fixed variance for all pixels.
        fixed_variance: [float]
           The fixed variance to use if learn_variance is False.
        """
        super(GaussianDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.learn_variance = learn_variance

        if not learn_variance:
            self.log_std = nn.Parameter(
                torch.ones(28, 28) * np.log(np.sqrt(fixed_variance)),
                requires_grad=False
            )

    def forward(self, z):
        """
        Given a batch of latent variables, return a Gaussian distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        h = self.decoder_net(z)

        if self.learn_variance:
            # h is flat (batch_size, 1568); split into two halves and reshape
            mean, log_std = torch.chunk(h, 2, dim=-1)
            mean = mean.view(-1, 28, 28)
            log_std = log_std.view(-1, 28, 28)
            std = torch.exp(log_std)
        else:
            mean = h
            std = torch.exp(self.log_std).unsqueeze(0).expand(h.shape[0], -1, -1)

        return td.Independent(td.Normal(loc=mean, scale=std), 2)


# =============================================================================
# VAE
# =============================================================================

class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """
    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module] 
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """
        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Uses KL divergence when available (Gaussian prior), otherwise falls back
        to the Monte Carlo estimate: E_q[log p(x|z) + log p(z) - log q(z|x)].

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
        """
        q = self.encoder(x)
        z = q.rsample()

        log_p_x_given_z = self.decoder(z).log_prob(x)

        # Try analytic KL; fall back to MC estimate for non-standard priors (e.g. MoG)
        try:
            kl = td.kl_divergence(q, self.prior())
            elbo = torch.mean(log_p_x_given_z - kl, dim=0)
        except NotImplementedError:
            log_p_z = self.prior().log_prob(z)
            log_q_z_given_x = q.log_prob(z)
            elbo = torch.mean(log_p_x_given_z + log_p_z - log_q_z_given_x, dim=0)

        return elbo

    def sample(self, n_samples=1, return_mean=False):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        return_mean: [bool]
           If True, return the mean of p(x|z) instead of sampling from it.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        decoder_dist = self.decoder(z)

        if return_mean:
            return decoder_dist.mean
        else:
            return decoder_dist.sample()

    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)


# =============================================================================
# Training & Evaluation
# =============================================================================

def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    model.train()

    total_steps = len(data_loader) * epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            x = x[0].to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(
                loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}"
            )
            progress_bar.update()


def evaluate(model, data_loader, device):
    """
    Evaluate the ELBO on a dataset.

    Parameters:
    model: [VAE]
       The VAE model to evaluate.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for evaluation.
    device: [torch.device]
        The device to use for evaluation.
    """
    model.eval()
    total_elbo = 0.0
    num_batches = 0
    with torch.no_grad():
        for x in data_loader:
            x = x[0].to(device)
            elbo_val = model.elbo(x)
            total_elbo += elbo_val.item()
            num_batches += 1
    avg_elbo = total_elbo / num_batches
    return avg_elbo


def plot_latent_space(model, data_loader, device, save_path='latent_space.png'):
    """
    Plot samples from the approximate posterior colored by class labels.
    For latent dimensions > 2, use PCA to project to 2D.

    Parameters:
    model: [VAE]
       The VAE model to use for encoding.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use (should include labels).
    device: [torch.device]
        The device to use.
    save_path: [str]
        Path to save the plot.
    """
    model.eval()
    latents = []
    labels = []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            q = model.encoder(x)
            z_mean = q.mean
            latents.append(z_mean.cpu().numpy())
            labels.append(y.numpy())

    latents = np.concatenate(latents, axis=0)
    labels = np.concatenate(labels, axis=0)

    if latents.shape[1] > 2:
        print(f"Latent dimension is {latents.shape[1]}, applying PCA to project to 2D...")
        pca = PCA(n_components=2)
        latents_2d = pca.fit_transform(latents)
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    else:
        latents_2d = latents

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        latents_2d[:, 0], latents_2d[:, 1],
        c=labels, cmap='tab10', alpha=0.6, s=5
    )
    plt.colorbar(scatter, label='Digit Class')
    plt.xlabel('First Component')
    plt.ylabel('Second Component')
    plt.title('Latent Space Visualization (Colored by Class)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Latent space plot saved to {save_path}")
    plt.close()
