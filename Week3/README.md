# Week 3 — Denoising Diffusion Probabilistic Model (DDPM)

Script: `ddpm.py`

Run all commands from the **project root**.

## Train

```bash
uv run Week3/ddpm.py train
```

## Sample from trained model

```bash
uv run Week3/ddpm.py sample --model outputs/models/ddpm_model.pt
```

## Test

```bash
uv run Week3/ddpm.py test --model outputs/models/ddpm_model.pt
```

## Key options

| Flag | Default | Description |
|---|---|---|
| `--data` | `tg` | `tg` (two Gaussians), `cb` (chequerboard), `mnist` |
| `--arch` | `fc` | `fc` (fully connected), `unet` |
| `--epochs` | `1` | Number of training epochs |
| `--lr` | `1e-3` | Learning rate |
| `--device` | `cpu` | `cpu`, `cuda`, `mps` |

Full option list: `uv run Week3/ddpm.py --help`
