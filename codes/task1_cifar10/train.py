from __future__ import annotations

import argparse
import csv
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer
from tqdm import tqdm

from data import DataLoaders, build_dataloaders
from model import ConfigurableCNN, count_parameters
from visualize import (
    plot_confusion_matrix,
    plot_first_layer_filters,
    plot_history,
    plot_misclassified,
    save_per_class_accuracy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a configurable CNN on CIFAR-10.")
    parser.add_argument("--experiment-name", default="baseline")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/task1"))
    parser.add_argument("--channels", default="64,128,256")
    parser.add_argument(
        "--activation", choices=["relu", "leaky_relu", "gelu", "silu"], default="relu"
    )
    parser.add_argument("--no-batch-norm", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--loss", choices=["cross_entropy", "focal"], default="cross_entropy")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--scheduler", choices=["none", "cosine", "multistep"], default="cosine"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--fake-data", action="store_true")
    parser.add_argument("--train-items", type=int, default=-1)
    parser.add_argument("--test-items", type=int, default=-1)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cross_entropy = F.cross_entropy(logits, targets, reduction="none")
        true_class_probability = torch.exp(-cross_entropy)
        return (((1.0 - true_class_probability) ** self.gamma) * cross_entropy).mean()


def make_criterion(args: argparse.Namespace) -> nn.Module:
    if args.loss == "focal":
        if args.label_smoothing != 0:
            raise ValueError("--label-smoothing is only supported with cross_entropy")
        return FocalLoss()
    return nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)


def make_optimizer(args: argparse.Namespace, model: nn.Module) -> Optimizer:
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
    optimizer_class = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
    return optimizer_class(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def make_scheduler(args: argparse.Namespace, optimizer: Optimizer):
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    if args.scheduler == "multistep":
        milestones = sorted({max(1, args.epochs // 2), max(2, 3 * args.epochs // 4)})
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.1
        )
    return None


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    progress = tqdm(loader, leave=False, desc="train" if training else "eval")

    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if scaler is not None
            else nullcontext()
        )
        with torch.set_grad_enabled(training), amp_context:
            logits = model(images)
            loss = criterion(logits, targets)

        if training:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size
        progress.set_postfix(loss=f"{total_loss / total_samples:.4f}")

    return total_loss / total_samples, 100.0 * total_correct / total_samples


@torch.inference_mode()
def detailed_evaluation(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, torch.Tensor, list[tuple[torch.Tensor, int, int]]]:
    model.eval()
    matrix = torch.zeros(10, 10, dtype=torch.int64)
    mistakes: list[tuple[torch.Tensor, int, int]] = []
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, targets in tqdm(loader, leave=False, desc="test"):
        device_images = images.to(device, non_blocking=True)
        device_targets = targets.to(device, non_blocking=True)
        logits = model(device_images)
        loss = criterion(logits, device_targets)
        predictions = logits.argmax(dim=1)

        total_loss += loss.item() * targets.size(0)
        total_correct += (predictions == device_targets).sum().item()
        total_samples += targets.size(0)
        for target, prediction in zip(targets, predictions.cpu()):
            matrix[int(target), int(prediction)] += 1
        if len(mistakes) < 20:
            wrong = predictions.cpu() != targets
            for image, target, prediction in zip(
                images[wrong], targets[wrong], predictions.cpu()[wrong]
            ):
                mistakes.append((image, int(target), int(prediction)))
                if len(mistakes) == 20:
                    break

    return (
        total_loss / total_samples,
        100.0 * total_correct / total_samples,
        matrix,
        mistakes,
    )


def save_history(history: list[dict[str, float]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channels = tuple(int(value) for value in args.channels.split(","))
    run_dir = args.output_dir / args.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)

    loaders: DataLoaders = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_size=args.val_size,
        seed=args.seed,
        augment=not args.no_augment,
        download=not args.no_download,
        fake_data=args.fake_data,
        train_items=args.train_items,
        test_items=args.test_items,
    )
    model = ConfigurableCNN(
        channels=channels,
        activation=args.activation,
        use_batch_norm=not args.no_batch_norm,
        dropout=args.dropout,
    ).to(device)
    criterion = make_criterion(args)
    optimizer = make_optimizer(args, model)
    scheduler = make_scheduler(args, optimizer)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None

    config = vars(args).copy()
    config.update(
        {
            "channels": channels,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
            "parameters": count_parameters(model),
        }
    )
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, default=str)

    print(
        f"Experiment: {args.experiment_name} | device: {config['device_name']} | "
        f"parameters: {config['parameters']:,}"
    )
    history: list[dict[str, float]] = []
    best_val_accuracy = -1.0
    best_epoch = 0
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, loaders.train, criterion, device, optimizer, scaler
        )
        val_loss, val_acc = run_epoch(model, loaders.val, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "learning_rate": current_lr,
            }
        )
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_accuracy": val_acc,
                },
                run_dir / "best.pt",
            )
        if scheduler is not None:
            scheduler.step()
        print(
            f"Epoch {epoch:03d}/{args.epochs}: train loss {train_loss:.4f}, "
            f"train acc {train_acc:.2f}%, val loss {val_loss:.4f}, val acc {val_acc:.2f}%"
        )

    elapsed_seconds = time.perf_counter() - start_time
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_acc, matrix, mistakes = detailed_evaluation(
        model, loaders.test, criterion, device
    )

    save_history(history, run_dir / "history.csv")
    plot_history(history, run_dir / "training_curves.png")
    plot_confusion_matrix(matrix, loaders.classes, run_dir / "confusion_matrix.png")
    save_per_class_accuracy(matrix, loaders.classes, run_dir / "per_class_accuracy.csv")
    plot_first_layer_filters(model, run_dir / "first_layer_filters.png")
    plot_misclassified(mistakes, loaders.classes, run_dir / "misclassified.png")

    summary = {
        "experiment_name": args.experiment_name,
        "parameters": count_parameters(model),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_error": 100.0 - test_acc,
        "training_seconds": elapsed_seconds,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
