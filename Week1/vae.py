# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Unified VAE script with configurable prior and decoder
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.utils import save_image
import argparse

from helpers import (
    GaussianPrior, MixtureOfGaussiansPrior,
    GaussianEncoder,
    BernoulliDecoder, GaussianDecoder,
    VAE,
    train, evaluate, plot_latent_space,
)


def get_data_loaders(decoder_type, batch_size):
    """
    Get MNIST data loaders with appropriate preprocessing.
    Bernoulli decoder -> binarized MNIST, Gaussian decoder -> continuous MNIST.
    """
    if decoder_type == 'bernoulli':
        threshold = 0.5
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (threshold < x).float().squeeze())
        ])
    else:  # gaussian
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.squeeze())
        ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/', train=False, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )
    return train_loader, test_loader


def build_model(args):
    """
    Build the VAE model from command-line arguments.
    """
    M = args.latent_dim

    # --- Prior ---
    if args.prior == 'gaussian':
        prior = GaussianPrior(M)
    elif args.prior == 'mog':
        prior = MixtureOfGaussiansPrior(M, K=args.mixture_components)

    # --- Encoder (always Gaussian) ---
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, M * 2),
    )
    encoder = GaussianEncoder(encoder_net)

    # --- Decoder ---
    if args.decoder == 'bernoulli':
        decoder_net = nn.Sequential(
            nn.Linear(M, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Unflatten(-1, (28, 28)),
        )
        decoder = BernoulliDecoder(decoder_net)

    elif args.decoder == 'gaussian':
        decoder_output_size = 784 * 2 if args.learn_variance else 784
        layers = [
            nn.Linear(M, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, decoder_output_size),
        ]
        # Only unflatten when not learning variance (single 28x28 mean output)
        if not args.learn_variance:
            layers.append(nn.Unflatten(-1, (28, 28)))
        decoder_net = nn.Sequential(*layers)
        decoder = GaussianDecoder(
            decoder_net,
            learn_variance=args.learn_variance,
            fixed_variance=args.fixed_variance,
        )

    model = VAE(prior, decoder, encoder)
    return model


def main():
    parser = argparse.ArgumentParser(description='VAE for MNIST')

    # Mode
    parser.add_argument('mode', type=str,
                        choices=['train', 'sample', 'sample_mean', 'evaluate', 'plot_latent'],
                        help='what to do when running the script')

    # Architecture
    parser.add_argument('--prior', type=str, default='gaussian',
                        choices=['gaussian', 'mog'],
                        help='prior distribution (default: %(default)s)')
    parser.add_argument('--decoder', type=str, default='bernoulli',
                        choices=['bernoulli', 'gaussian'],
                        help='decoder/output distribution (default: %(default)s)')

    # File paths
    parser.add_argument('--model', type=str, default='outputs/models/model.pt',
                        help='file to save/load model (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='outputs/images/samples.png',
                        help='file to save samples (default: %(default)s)')
    parser.add_argument('--latent-plot', type=str, default='outputs/images/latent_space.png',
                        help='file to save latent space plot (default: %(default)s)')

    # Training
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda', 'mps'],
                        help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='batch size (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=10, metavar='N',
                        help='dimension of latent variable (default: %(default)s)')

    # MoG-specific
    parser.add_argument('--mixture-components', type=int, default=10, metavar='K',
                        help='number of mixture components for MoG prior (default: %(default)s)')

    # Gaussian decoder-specific
    parser.add_argument('--learn-variance', action='store_true',
                        help='learn per-pixel variance (only for gaussian decoder)')
    parser.add_argument('--fixed-variance', type=float, default=0.1, metavar='V',
                        help='fixed variance when not learning it (default: %(default)s)')

    args = parser.parse_args()

    # Build prefix from architecture choices
    prefix = f"{args.prior}_{args.decoder}"
    if args.decoder == 'gaussian':
        prefix += '_learnvar' if args.learn_variance else f'_fixvar{args.fixed_variance}'
    if args.prior == 'mog':
        prefix += f'_K{args.mixture_components}'

    # Prefix output filenames
    for attr in ['model', 'samples', 'latent_plot']:
        path = getattr(args, attr)
        d, name = os.path.split(path)
        setattr(args, attr, os.path.join(d, f"{prefix}_{name}"))

    # Print options
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(f'  {key} = {value}')

    # Ensure output directories exist
    for path in [args.model, args.samples, args.latent_plot]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    device = args.device

    # Data
    train_loader, test_loader = get_data_loaders(args.decoder, args.batch_size)

    # Model
    model = build_model(args).to(device)

    # ---- Modes ----
    if args.mode == 'train':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(model, optimizer, train_loader, args.epochs, device)
        torch.save(model.state_dict(), args.model)
        print(f"Model saved to {args.model}")

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(device)))
        model.eval()
        with torch.no_grad():
            samples = model.sample(64, return_mean=False).cpu()
            samples = torch.clamp(samples, 0, 1)
            save_image(samples.view(64, 1, 28, 28), args.samples)
            print(f"Samples saved to {args.samples}")

    elif args.mode == 'sample_mean':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(device)))
        model.eval()
        with torch.no_grad():
            samples = model.sample(64, return_mean=True).cpu()
            samples = torch.clamp(samples, 0, 1)
            save_image(samples.view(64, 1, 28, 28), args.samples)
            print(f"Sample means saved to {args.samples}")

    elif args.mode == 'evaluate':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(device)))
        avg_elbo = evaluate(model, test_loader, device)
        print(f'Average ELBO on test set: {avg_elbo:.4f}')

    elif args.mode == 'plot_latent':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(device)))
        plot_latent_space(model, test_loader, device, args.latent_plot)


if __name__ == '__main__':
    main()
