from __future__ import annotations

import csv
import json
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from tqdm import tqdm

from data import Loaders
from models import VGGA, count_parameters


@dataclass
class TrainResult:
    history: list[dict[str, float]]
    step_metrics: list[dict[str, float]]
    best_val_accuracy: float
    best_epoch: int
    test_loss: float
    test_accuracy: float
    elapsed_seconds: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_csv(rows: list[dict[str, float]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    samples = 0
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss_sum += loss.item() * targets.size(0)
            correct += (logits.argmax(1) == targets).sum().item()
            samples += targets.size(0)
    return loss_sum / samples, 100.0 * correct / samples


def train_model(
    *,
    model: VGGA,
    loaders: Loaders,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    optimizer_name: str,
    momentum: float,
    weight_decay: float,
    output_dir: Path,
    use_amp: bool,
    max_steps: int = -1,
    record_steps: bool = True,
) -> TrainResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    if optimizer_name == "sgd":
        optimizer: Optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    scaler = (
        torch.amp.GradScaler("cuda", enabled=True)
        if use_amp and device.type == "cuda"
        else None
    )

    history: list[dict[str, float]] = []
    step_metrics: list[dict[str, float]] = []
    best_val_accuracy = -1.0
    best_epoch = 0
    global_step = 0
    previous_gradient: torch.Tensor | None = None
    previous_weight: torch.Tensor | None = None
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0
        progress = tqdm(loaders.train, leave=False, desc=f"epoch {epoch}/{epochs}")
        for images, targets in progress:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            amp_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if scaler is not None
                else nullcontext()
            )
            with amp_context:
                logits = model(images)
                loss = criterion(logits, targets)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()

            if record_steps:
                gradient = model.tracked_weight.grad.detach().float().cpu().flatten()
                weight = model.tracked_weight.detach().float().cpu().flatten()
                gradient_norm = gradient.norm().item()
                gradient_change = (
                    (gradient - previous_gradient).norm().item()
                    if previous_gradient is not None
                    else 0.0
                )
                parameter_distance = (
                    (weight - previous_weight).norm().item()
                    if previous_weight is not None
                    else 0.0
                )
                cosine_similarity = (
                    torch.nn.functional.cosine_similarity(
                        gradient.unsqueeze(0), previous_gradient.unsqueeze(0)
                    ).item()
                    if previous_gradient is not None
                    else 1.0
                )
                change_per_distance = (
                    gradient_change / parameter_distance
                    if parameter_distance > 1e-12
                    else float("nan")
                )
                step_metrics.append(
                    {
                        "step": global_step,
                        "epoch": epoch,
                        "loss": loss.item(),
                        "gradient_norm": gradient_norm,
                        "gradient_change": gradient_change,
                        "parameter_distance": parameter_distance,
                        "gradient_change_per_distance": change_per_distance,
                        "gradient_cosine_similarity": cosine_similarity,
                    }
                )
                previous_gradient = gradient
                previous_weight = weight

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            batch_size = targets.size(0)
            epoch_loss += loss.item() * batch_size
            epoch_correct += (logits.argmax(1) == targets).sum().item()
            epoch_samples += batch_size
            global_step += 1
            progress.set_postfix(loss=f"{epoch_loss / epoch_samples:.4f}")
            if max_steps > 0 and global_step >= max_steps:
                break

        train_loss = epoch_loss / epoch_samples
        train_accuracy = 100.0 * epoch_correct / epoch_samples
        val_loss, val_accuracy = evaluate(model, loaders.val, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "batch_norm": model.batch_norm,
                    "epoch": epoch,
                    "val_accuracy": val_accuracy,
                },
                output_dir / "best.pt",
            )
        print(
            f"epoch {epoch:02d}: train loss={train_loss:.4f}, "
            f"train acc={train_accuracy:.2f}%, val loss={val_loss:.4f}, "
            f"val acc={val_accuracy:.2f}%"
        )
        if max_steps > 0 and global_step >= max_steps:
            break

    elapsed = time.perf_counter() - started
    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_accuracy = evaluate(model, loaders.test, criterion, device)
    save_csv(history, output_dir / "history.csv")
    save_csv(step_metrics, output_dir / "steps.csv")
    summary = {
        "batch_norm": model.batch_norm,
        "parameters": count_parameters(model),
        "learning_rate": learning_rate,
        "optimizer": optimizer_name,
        "epochs_completed": len(history),
        "steps_completed": global_step,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "elapsed_seconds": elapsed,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return TrainResult(
        history,
        step_metrics,
        best_val_accuracy,
        best_epoch,
        test_loss,
        test_accuracy,
        elapsed,
    )
