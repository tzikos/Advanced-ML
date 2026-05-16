# Week 1 — Variational Autoencoder (MNIST)

Script: `vae.py`

Run all commands from the **project root**.

## Train

```bash
uv run Week1/vae.py train
```

## Sample from model

```bash
uv run Week1/vae.py sample --model outputs/models/model.pt
```

## Evaluate (ELBO on test set)

```bash
uv run Week1/vae.py evaluate --model outputs/models/model.pt
```

## Plot latent space

```bash
uv run Week1/vae.py plot_latent --model outputs/models/model.pt
```

## Key options

| Flag | Default | Description |
|---|---|---|
| `--prior` | `gaussian` | `gaussian`, `mog`, `flow` |
| `--decoder` | `bernoulli` | `bernoulli`, `gaussian` |
| `--epochs` | `10` | Number of training epochs |
| `--latent-dim` | `10` | Latent space dimensionality |
| `--device` | `cpu` | `cpu`, `cuda`, `mps` |

Full option list: `uv run Week1/vae.py --help`
