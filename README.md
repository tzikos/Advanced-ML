# Advanced Machine Learning - Exercises

This repository contains exercises and implementations for the DTU course 02460 (Advanced Machine Learning Spring).

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management.

```bash
uv pip install torch torchvision tqdm matplotlib scikit-learn
```

Or sync from `pyproject.toml`:

```bash
uv sync
```

## Week 1: Variational Autoencoders (VAE)

A unified VAE implementation with configurable prior and decoder.

### Architecture Options

| Flag | Choices | Description |
|------|---------|-------------|
| `--prior` | `gaussian` (default), `mog` | Prior distribution over latent space |
| `--decoder` | `bernoulli` (default), `gaussian` | Output/decoder distribution |
| `--learn-variance` | flag | Learn per-pixel variance (gaussian decoder only) |
| `--fixed-variance` | float (default: 0.1) | Fixed variance value (gaussian decoder only) |
| `--mixture-components` | int (default: 10) | Number of components (MoG prior only) |

### Modes

| Mode | Description |
|------|-------------|
| `train` | Train the model and save weights |
| `evaluate` | Compute average ELBO on test set |
| `sample` | Generate samples from p(x\|z) |
| `sample_mean` | Show mean of p(x\|z) instead of sampling |
| `plot_latent` | Visualise latent space colored by class (PCA for dim > 2) |

### Examples

#### 1. Gaussian prior + Bernoulli decoder (binarized MNIST)

```bash
# Train
python Week1/vae.py train \
    --prior gaussian --decoder bernoulli \
    --latent-dim 10 --epochs 5 --batch-size 128 \
    --model outputs/models/model.pt

# Evaluate
python Week1/vae.py evaluate \
    --prior gaussian --decoder bernoulli \
    --latent-dim 10 --batch-size 128 \
    --model outputs/models/model.pt

# Sample
python Week1/vae.py sample \
    --prior gaussian --decoder bernoulli \
    --latent-dim 10 \
    --model outputs/models/model.pt \
    --samples outputs/images/samples.png

# Plot latent space
python Week1/vae.py plot_latent \
    --prior gaussian --decoder bernoulli \
    --latent-dim 10 --batch-size 128 \
    --model outputs/models/model.pt \
    --latent-plot outputs/images/latent_space.png
```

#### 2. MoG prior + Bernoulli decoder (binarized MNIST)

```bash
# Train
python Week1/vae.py train \
    --prior mog --decoder bernoulli \
    --latent-dim 10 --mixture-components 10 \
    --epochs 5 --batch-size 128 \
    --model outputs/models/model_mog.pt

# Evaluate
python Week1/vae.py evaluate \
    --prior mog --decoder bernoulli \
    --latent-dim 10 --mixture-components 10 \
    --batch-size 128 \
    --model outputs/models/model_mog.pt

# Plot latent space
python Week1/vae.py plot_latent \
    --prior mog --decoder bernoulli \
    --latent-dim 10 --mixture-components 10 \
    --batch-size 128 \
    --model outputs/models/model_mog.pt \
    --latent-plot outputs/images/latent_space_mog.png
```

#### 3. Gaussian prior + Gaussian decoder (continuous MNIST)

```bash
# Train with learned variance
python Week1/vae.py train \
    --prior gaussian --decoder gaussian --learn-variance \
    --latent-dim 10 --epochs 5 --batch-size 128 \
    --model outputs/models/model_gaussian_learned.pt

# Train with fixed variance
python Week1/vae.py train \
    --prior gaussian --decoder gaussian --fixed-variance 0.1 \
    --latent-dim 10 --epochs 5 --batch-size 128 \
    --model outputs/models/model_gaussian_fixed.pt

# Sample from p(x|z)
python Week1/vae.py sample \
    --prior gaussian --decoder gaussian --learn-variance \
    --latent-dim 10 \
    --model outputs/models/model_gaussian_learned.pt \
    --samples outputs/images/samples_gaussian.png

# Show mean of p(x|z) (often looks better)
python Week1/vae.py sample_mean \
    --prior gaussian --decoder gaussian --learn-variance \
    --latent-dim 10 \
    --model outputs/models/model_gaussian_learned.pt \
    --samples outputs/images/samples_mean_gaussian.png
```

### All Options

| Option | Default | Description |
|--------|---------|-------------|
| `--device` | `cpu` | Device: `cpu`, `cuda`, or `mps` |
| `--latent-dim` | `10` | Dimension of latent space |
| `--epochs` | `10` | Number of training epochs |
| `--batch-size` | `128` | Batch size |
| `--model` | `outputs/models/model.pt` | Path to save/load model weights |
| `--samples` | `outputs/images/samples.png` | Path to save generated samples |
| `--latent-plot` | `outputs/images/latent_space.png` | Path to save latent space plot |
| `--prior` | `gaussian` | Prior: `gaussian` or `mog` |
| `--decoder` | `bernoulli` | Decoder: `bernoulli` or `gaussian` |
| `--mixture-components` | `10` | MoG components (only with `--prior mog`) |
| `--learn-variance` | off | Learn per-pixel variance (only with `--decoder gaussian`) |
| `--fixed-variance` | `0.1` | Fixed variance (only with `--decoder gaussian`) |

> **Note:** When evaluating/sampling, ensure `--prior`, `--decoder`, `--latent-dim`, and other architecture options match those used during training.