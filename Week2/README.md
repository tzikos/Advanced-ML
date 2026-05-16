# Week 2 — Normalizing Flows

Script: `flow.py`

Run all commands from the **project root**.

## Train

```bash
uv run Week2/flow.py train
```

## Sample from trained model

```bash
uv run Week2/flow.py sample --model outputs/models/model.pt
```

## Key options

| Flag | Default | Description |
|---|---|---|
| `--data` | `tg` | `tg` (two Gaussians), `cb` (chequerboard), `mnist` |
| `--mask` | `half` | `half`, `random`, `chequerboard` |
| `--epochs` | `1` | Number of training epochs |
| `--lr` | `1e-3` | Learning rate |
| `--device` | `cpu` | `cpu`, `cuda`, `mps` |

Full option list: `uv run Week2/flow.py --help`
