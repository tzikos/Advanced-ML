# Week 10 — Graph Neural Network (MUTAG Classification)

Scripts: `gnn_graph_classification.py` (main exercise), `tune_gnn_optuna.py` (hyperparameter tuning), and `week10_exercises.ipynb` (notebook).

Run all commands from the **project root**.

## Run the main script

```bash
uv run Week10/gnn_graph_classification.py
```

## Run hyperparameter tuning (Optuna)

```bash
uv run Week10/tune_gnn_optuna.py --trials 40 --device cpu --save-model
```

## Run the notebook

```bash
uv run jupyter notebook Week10/week10_exercises.ipynb
```

Or open `week10_exercises.ipynb` directly in VS Code.

---

This diagram is based on the `SimpleGNN` class in `gnn_graph_classification.py`, using the components from `__init__` and the computation in `forward`.

## Model components (`__init__`)

- **Input network**: `Linear(node_feature_dim → state_dim) + ReLU`
- **Message networks**: `num_message_passing_rounds` blocks of `Linear(state_dim → state_dim) + ReLU`
- **Update networks**: `num_message_passing_rounds` blocks of `Linear(state_dim → state_dim) + ReLU`
- **Output network**: `Linear(state_dim → 1)`

## Forward pass architecture (`forward`)

### Preview-safe diagram (always visible)

```text
┌───────────────────────────────┐
│ Node features x               │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────────────────┐
│ Input Net                                 │
│ Linear(node_feature_dim→state_dim) + ReLU │
└───────────────┬───────────────────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Initial node states h(0)      │
└───────────────┬───────────────┘
                │
                ▼
      Repeat for r = 0 ... R-1
                │
                ▼
┌───────────────────────────────┐
│ Message Net r                 │
│ m = M_r(h)                    │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Edge gather + index_add       │
│ a_i = Σ incoming messages     │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Update Net r                  │
│ u = U_r(a)                    │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Residual update               │
│ h ← h + u                     │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Graph readout (index_add)     │
│ graph_state = Σ node states   │
│ per graph via batch index     │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Output Net                    │
│ Linear(state_dim→1)           │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Logit per graph               │
└───────────────────────────────┘
```

### Mermaid diagram (optional)

```mermaid
flowchart TD
    A[Node features x] --> B[Input Net\nLinear + ReLU]
    B --> C[state h(0)]

    subgraph MP[Message Passing rounds r = 0 ... R-1]
      C --> D[Message Net r\nm = M_r(h)]
      D --> E[Edge gather\nm[edge_index[0]]]
      E --> F[index_add aggregation\na_i = sum of incoming messages]
      F --> G[Update Net r\nU_r(a)]
      G --> H[Residual update\nh <- h + U_r(a)]
      H --> C
    end

    C --> I[Graph readout\nSum over nodes by batch\ngraph_state = index_add(batch, h)]
    I --> J[Output Net\nLinear(state_dim -> 1)]
    J --> K[Logit per graph]
```

## Tensor flow summary

1. Initialize node state: `h = input_net(x)`
2. For each message-passing round `r`:
   - `m = message_net[r](h)`
   - Aggregate incoming messages with `index_add` using `edge_index`
   - Update with residual connection: `h = h + update_net[r](aggregated)`
3. Pool node states to graph states with `index_add(..., batch, h)`
4. Produce one graph-level logit with `output_net(graph_state)`

## Hyperparameter tuning for Question C.4 (Optuna)

Use the tuner script to optimize validation loss without touching the test set:

```bash
uv run python Week10/tune_gnn_optuna.py --trials 40 --device cpu --save-model
```

Run only the original update architecture:

```bash
uv run python Week10/tune_gnn_optuna.py --trials 100 --device cpu --save-model --update-types residual_mlp
```

What it tunes:

- `state_dim`
- `num_message_passing_rounds`
- `update_type` (`residual_mlp`, `gru`, `gated_residual`, `replace_mlp`)
- `dropout`
- `lr`
- `weight_decay`
- `scheduler_eta_min_ratio` (for `CosineAnnealingLR`)
- `epochs`

Artifacts are saved in `Week10/outputs/`:

- `optuna_best_config.json`
- `best_gnn_optuna.pt` (if `--save-model` is enabled)
