import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import optuna
import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader


class SimpleGNN(torch.nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        state_dim: int,
        num_message_passing_rounds: int,
        dropout: float,
        update_type: str,
    ):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.state_dim = state_dim
        self.num_message_passing_rounds = num_message_passing_rounds
        self.dropout = dropout
        self.update_type = update_type

        valid_update_types = {"residual_mlp", "gru", "gated_residual", "replace_mlp"}
        if self.update_type not in valid_update_types:
            raise ValueError(f"Unsupported update_type '{self.update_type}'. Must be one of {valid_update_types}.")

        self.input_net = torch.nn.Sequential(
            torch.nn.Linear(self.node_feature_dim, self.state_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.dropout),
        )

        self.message_net = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(self.state_dim, self.state_dim),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(p=self.dropout),
                )
                for _ in range(num_message_passing_rounds)
            ]
        )

        self.update_net = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(self.state_dim, self.state_dim),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(p=self.dropout),
                )
                for _ in range(num_message_passing_rounds)
            ]
        )

        self.gru_update = torch.nn.ModuleList(
            [torch.nn.GRUCell(self.state_dim, self.state_dim) for _ in range(num_message_passing_rounds)]
        )

        self.gate_net = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(self.state_dim, self.state_dim),
                    torch.nn.Sigmoid(),
                )
                for _ in range(num_message_passing_rounds)
            ]
        )

        self.output_net = torch.nn.Linear(self.state_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        num_graphs = batch.max() + 1
        num_nodes = batch.shape[0]

        state = self.input_net(x)

        for r in range(self.num_message_passing_rounds):
            message = self.message_net[r](state)
            aggregated = x.new_zeros((num_nodes, self.state_dim))
            aggregated = aggregated.index_add(0, edge_index[1], message[edge_index[0]])

            if self.update_type == "residual_mlp":
                state = state + self.update_net[r](aggregated)
            elif self.update_type == "gru":
                state = self.gru_update[r](aggregated, state)
            elif self.update_type == "gated_residual":
                delta = self.update_net[r](aggregated)
                gate = self.gate_net[r](aggregated)
                state = state + gate * delta
            elif self.update_type == "replace_mlp":
                state = self.update_net[r](aggregated)

        graph_state = x.new_zeros((num_graphs, self.state_dim))
        graph_state = torch.index_add(graph_state, 0, batch, state)
        out = self.output_net(graph_state).flatten()
        return out


def make_loaders(data_root: str, seed: int):
    dataset = TUDataset(root=data_root, name="MUTAG")
    node_feature_dim = dataset.num_features

    rng = torch.Generator().manual_seed(seed)
    train_dataset, validation_dataset, test_dataset = random_split(dataset, (100, 44, 44), generator=rng)

    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=44, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=44, shuffle=False)
    return train_loader, validation_loader, test_loader, node_feature_dim


def evaluate_loader_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_graphs = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y.float())
            total_loss += loss.item() * data.num_graphs
            total_correct += ((out > 0) == data.y).sum().item()
            total_graphs += data.num_graphs
    avg_loss = total_loss / len(loader.dataset)
    accuracy = total_correct / total_graphs
    return avg_loss, accuracy


def objective(trial: optuna.trial.Trial, args: argparse.Namespace) -> float:
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    train_loader, validation_loader, _, node_feature_dim = make_loaders(args.data_root, args.seed)

    state_dim = trial.suggest_categorical("state_dim", [8, 16, 32, 64, 128])
    rounds = trial.suggest_int("num_message_passing_rounds", 2, 8)
    update_type = trial.suggest_categorical("update_type", args.update_types)
    dropout = trial.suggest_float("dropout", 0.0, 0.6)
    lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True)
    eta_min_ratio = trial.suggest_float("scheduler_eta_min_ratio", 1e-4, 5e-1, log=True)
    epochs = trial.suggest_int("epochs", 80, 600)
    eta_min = lr * eta_min_ratio

    model = SimpleGNN(
        node_feature_dim,
        state_dim,
        rounds,
        dropout=dropout,
        update_type=update_type,
    ).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=eta_min,
    )

    for epoch in range(epochs):
        model.train()
        for data in train_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        val_loss, _ = evaluate_loader_metrics(model, validation_loader, criterion, device)
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_loss


def retrain_best_and_validate(best_params: dict, args: argparse.Namespace) -> tuple[float, dict[str, str]]:
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    train_loader, validation_loader, test_loader, node_feature_dim = make_loaders(args.data_root, args.seed)

    model = SimpleGNN(
        node_feature_dim,
        best_params["state_dim"],
        best_params["num_message_passing_rounds"],
        dropout=best_params["dropout"],
        update_type=best_params["update_type"],
    ).to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    eta_min = best_params["lr"] * best_params["scheduler_eta_min_ratio"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=best_params["epochs"],
        eta_min=eta_min,
    )

    train_losses: list[float] = []
    train_accuracies: list[float] = []
    validation_losses: list[float] = []
    validation_accuracies: list[float] = []

    for _ in range(best_params["epochs"]):
        model.train()
        for data in train_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        train_loss, train_accuracy = evaluate_loader_metrics(model, train_loader, criterion, device)
        validation_loss, validation_accuracy = evaluate_loader_metrics(model, validation_loader, criterion, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_accuracy)

    val_loss = validation_losses[-1]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loss_plot_path = output_dir / "best_hparams_loss.png"
    plt.figure(figsize=(8, 4.5))
    plt.plot(train_losses, label="Train")
    plt.plot(validation_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("BCE loss")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_plot_path, dpi=150)
    plt.close()

    accuracy_plot_path = output_dir / "best_hparams_accuracy.png"
    plt.figure(figsize=(8, 4.5))
    plt.plot(train_accuracies, label="Train")
    plt.plot(validation_accuracies, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(accuracy_plot_path, dpi=150)
    plt.close()

    if args.save_model:
        model_path = output_dir / "best_gnn_optuna.pt"
        torch.save(model.state_dict(), model_path)

    with torch.no_grad():
        model.eval()
        test_data = next(iter(test_loader)).to(device)
        test_predictions = model(test_data.x, test_data.edge_index, test_data.batch).cpu()
        torch.save(test_predictions, args.test_predictions_path)

    plot_paths = {
        "loss_plot": str(loss_plot_path),
        "accuracy_plot": str(accuracy_plot_path),
    }
    return val_loss, plot_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna tuning for Week10 MUTAG GNN (validation-loss only)")
    parser.add_argument("--trials", type=int, default=40, help="Number of Optuna trials")
    parser.add_argument("--seed", type=int, default=0, help="Seed used for split and initialization")
    parser.add_argument("--data-root", type=str, default="./Week10/data", help="Root for TUDataset storage")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Training device")
    parser.add_argument("--output-dir", type=str, default="./Week10/outputs", help="Directory for artifacts")
    parser.add_argument("--save-model", action="store_true", help="Save best model state_dict after retraining")
    parser.add_argument("--study-name", type=str, default="week10_mutag_gnn_optuna", help="Optuna study name")
    parser.add_argument(
        "--update-types",
        type=str,
        nargs="+",
        default=["residual_mlp", "gru", "gated_residual", "replace_mlp"],
        help=(
            "Allowed node update types for Optuna search. "
            "Choices: residual_mlp gru gated_residual replace_mlp"
        ),
    )
    parser.add_argument(
        "--test-predictions-path",
        type=str,
        default="test_predictions.pt",
        help="Path to save test-set predictions from the best retrained model",
    )
    args = parser.parse_args()

    valid_update_types = {"residual_mlp", "gru", "gated_residual", "replace_mlp"}
    invalid = set(args.update_types) - valid_update_types
    if invalid:
        parser.error(
            f"Invalid --update-types values: {sorted(invalid)}. "
            f"Valid options are: {sorted(valid_update_types)}"
        )

    args.update_types = list(dict.fromkeys(args.update_types))
    return args


def main() -> None:
    args = parse_args()

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=args.study_name,
    )

    study.optimize(lambda trial: objective(trial, args), n_trials=args.trials)

    best_params = study.best_trial.params
    best_val_loss_retrain, plot_paths = retrain_best_and_validate(best_params, args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "optuna_best_config.json"

    summary = {
        "best_trial_number": study.best_trial.number,
        "best_trial_value": study.best_trial.value,
        "best_params": best_params,
        "scheduler": "CosineAnnealingLR",
        "retrained_validation_loss": best_val_loss_retrain,
        "plot_paths": plot_paths,
        "test_predictions_path": args.test_predictions_path,
        "note": "No test set used during tuning or final retrain.",
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Best trial:")
    print(f"  number: {study.best_trial.number}")
    print(f"  validation loss: {study.best_trial.value:.6f}")
    print(f"  params: {study.best_trial.params}")
    print(f"Retrained validation loss: {best_val_loss_retrain:.6f}")
    print(f"Loss plot: {plot_paths['loss_plot']}")
    print(f"Accuracy plot: {plot_paths['accuracy_plot']}")
    print(f"Test predictions: {args.test_predictions_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
