# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-02-11)

import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
from tqdm import tqdm


class DDPM(nn.Module):
    def __init__(self, network, beta_1=1e-4, beta_T=2e-2, T=100):
        """
        Initialize a DDPM model.

        Parameters:
        network: [nn.Module]
            The network to use for the diffusion process.
        beta_1: [float]
            The noise at the first step of the diffusion process.
        beta_T: [float]
            The noise at the last step of the diffusion process.
        T: [int]
            The number of steps in the diffusion process.
        """
        super(DDPM, self).__init__()
        self.network = network
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T

        self.beta = nn.Parameter(torch.linspace(beta_1, beta_T, T), requires_grad=False)
        self.alpha = nn.Parameter(1 - self.beta, requires_grad=False)
        self.alpha_cumprod = nn.Parameter(self.alpha.cumprod(dim=0), requires_grad=False)
    
    def negative_elbo(self, x):
        """
        Evaluate the DDPM negative ELBO on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The negative ELBO of the batch of dimension `(batch_size,)`.
        """

        ### Implement Algorithm 1 here ###
        batch_size = x.shape[0]
        t = torch.randint(0, self.T, (batch_size,), device=x.device)
        t_norm = t.float().view(-1, 1) / max(self.T - 1, 1)
        noise = torch.randn_like(x)
        x_t = self.alpha_cumprod[t].sqrt().view(-1, 1) * x + (1 - self.alpha_cumprod[t]).sqrt().view(-1, 1) * noise
        pred_noise = self.network(x_t, t_norm)
        neg_elbo = F.mse_loss(pred_noise, noise)

        return neg_elbo

    def sample(self, shape):
        """
        Sample from the model.

        Parameters:
        shape: [tuple]
            The shape of the samples to generate.
        Returns:
        [torch.Tensor]
            The generated samples.
        """
        # Sample x_t for t=T (i.e., Gaussian noise)
        x_t = torch.randn(shape).to(self.alpha.device)

        # Sample x_t given x_{t+1} until x_0 is sampled
        for t in range(self.T-1, -1, -1):
            ### Implement the remaining of Algorithm 2 here ###
            t_tensor = torch.full((shape[0], 1), float(t), device=self.alpha.device)
            t_norm = t_tensor / max(self.T - 1, 1)
            pred_noise = self.network(x_t, t_norm)
            x_t = (x_t - self.beta[t] * pred_noise / (1 - self.alpha_cumprod[t]).sqrt()) / self.alpha[t].sqrt()
            x_t += torch.randn_like(x_t) * (self.beta[t] / self.alpha[t]).sqrt() if t > 0 else 0

        return x_t

    def loss(self, x):
        """
        Evaluate the DDPM loss on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The loss for the batch.
        """
        return self.negative_elbo(x).mean()


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a Flow model.

    Parameters:
    model: [Flow]
       The model to train.
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

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(device)
            optimizer.zero_grad()
            loss = model.loss(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()


class FcNetwork(nn.Module):
    def __init__(self, input_dim, num_hidden, T):
        """
        Initialize a fully connected network for the DDPM, where the forward function also take time as an argument.
        
        parameters:
        input_dim: [int]
            The dimension of the input data.
        num_hidden: [int]
            The number of hidden units in the network.
        """
        super(FcNetwork, self).__init__()
        self.T = T
        self.network = nn.Sequential(nn.Linear(input_dim+1, num_hidden), nn.ReLU(), 
                                     nn.Linear(num_hidden, num_hidden), nn.ReLU(), 
                                     nn.Linear(num_hidden, input_dim))

    def forward(self, x, t):
        """"
        Forward function for the network.
        
        parameters:
        x: [torch.Tensor]
            The input data of dimension `(batch_size, input_dim)`
        t: [torch.Tensor]
            The time steps to use for the forward pass of dimension `(batch_size, 1)`
        """
        x_t_cat = torch.cat([x, t], dim=1)
        return self.network(x_t_cat)


if __name__ == "__main__":
    import os
    import sys
    import torch.utils.data
    from torchvision import datasets, transforms
    from torchvision.utils import save_image

    # Allow running as: python Week3/ddpm.py ...
    # by adding the repository root to sys.path.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import Week2.ToyData as ToyData

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'test'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--data', type=str, default='tg', choices=['tg', 'cb', 'mnist'], help='dataset to use {tg: two Gaussians, cb: chequerboard} (default: %(default)s)')
    parser.add_argument('--model', type=str, default='outputs/models/ddpm_model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='outputs/images/ddpm_samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='V', help='learning rate for training (default: %(default)s)')
    parser.add_argument('--arch', type=str, default='fc', choices=['fc', 'unet'], help='network architecture to use (default: %(default)s)')

    args = parser.parse_args()

    def add_suffix(path, data_name, arch_name):
        root, ext = os.path.splitext(path)
        suffix = f"_{data_name}_{arch_name}"
        if root.endswith(suffix):
            return path
        return f"{root}{suffix}{ext}"

    args.model = add_suffix(args.model, args.data, args.arch)
    args.samples = add_suffix(args.samples, args.data, args.arch)

    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    # Generate/load the data
    is_mnist = (args.data == 'mnist')
    toy = None

    if is_mnist:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x + torch.rand(x.shape) / 255),
            transforms.Lambda(lambda x: (x - 0.5) * 2.0),
            transforms.Lambda(lambda x: x.flatten()),
        ])

        train_data = datasets.MNIST(
            'data/',
            train=True,
            download=True,
            transform=transform,
        )
        test_data = datasets.MNIST(
            'data/',
            train=False,
            download=True,
            transform=transform,
        )

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
        D = 28 * 28
    else:
        n_data = 10000000
        toy = {'tg': ToyData.TwoGaussians, 'cb': ToyData.Chequerboard}[args.data]()
        transform = lambda x: (x - 0.5) * 2.0
        train_loader = torch.utils.data.DataLoader(transform(toy().sample((n_data,))), batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(transform(toy().sample((n_data,))), batch_size=args.batch_size, shuffle=True)
        D = next(iter(train_loader)).shape[1]

    # Define the network
    num_hidden = 128

    # Set the number of steps in the diffusion process
    T = 1000

    if args.arch == 'unet':
        if not is_mnist:
            raise ValueError("U-Net architecture is only supported with --data mnist")
        from Week3.unet import Unet
        network = Unet()
    else:
        network = FcNetwork(D, num_hidden, T=T)

    # Define model
    model = DDPM(network, T=T).to(args.device)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Train model
        train(model, optimizer, train_loader, args.epochs, args.device)

        # Save model
        os.makedirs(os.path.dirname(args.model), exist_ok=True)
        torch.save(model.state_dict(), args.model)

    elif args.mode == 'sample':
        import matplotlib.pyplot as plt
        import numpy as np

        # Load the model
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            n_samples = 64 if is_mnist else 10000
            samples = (model.sample((n_samples, D))).cpu()

        # Transform the samples back to the original space
        samples = samples /2 + 0.5

        os.makedirs(os.path.dirname(args.samples), exist_ok=True)

        if is_mnist:
            samples = samples.clamp(0.0, 1.0).reshape(-1, 1, 28, 28)
            save_image(samples, args.samples, nrow=8)
        else:
            # Plot the density of the toy data and the model samples
            coordinates = [[[x,y] for x in np.linspace(*toy.xlim, 1000)] for y in np.linspace(*toy.ylim, 1000)]
            prob = torch.exp(toy().log_prob(torch.tensor(coordinates)))

            fig, ax = plt.subplots(1, 1, figsize=(7, 5))
            im = ax.imshow(prob, extent=[toy.xlim[0], toy.xlim[1], toy.ylim[0], toy.ylim[1]], origin='lower', cmap='YlOrRd')
            ax.scatter(samples[:, 0], samples[:, 1], s=1, c='black', alpha=0.5)
            ax.set_xlim(toy.xlim)
            ax.set_ylim(toy.ylim)
            ax.set_aspect('equal')
            fig.colorbar(im)
            plt.savefig(args.samples)
            plt.close()

    elif args.mode == 'test':
        # Load the model
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        model.eval()

        with torch.no_grad():
            losses = []
            for x in test_loader:
                if isinstance(x, (list, tuple)):
                    x = x[0]
                x = x.to(args.device)
                losses.append(model.loss(x).item())
        print(f"test_loss = {sum(losses)/len(losses):.6f}")
