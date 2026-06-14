from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from data import CIFAR10_MEAN, CIFAR10_STD


def plot_history(history: list[dict[str, float]], output_path: Path) -> None:
    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, [row["train_acc"] for row in history], label="Train")
    axes[1].plot(epochs, [row["val_acc"] for row in history], label="Validation")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy (%)", title="Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_confusion_matrix(
    matrix: torch.Tensor, classes: tuple[str, ...], output_path: Path
) -> None:
    matrix_np = matrix.numpy()
    row_sums = matrix_np.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix_np,
        row_sums,
        out=np.zeros_like(matrix_np, dtype=float),
        where=row_sums != 0,
    )
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    figure.colorbar(image, ax=axis, fraction=0.046)
    axis.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted label",
        ylabel="True label",
        title="Normalized confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_per_class_accuracy(
    matrix: torch.Tensor, classes: tuple[str, ...], output_path: Path
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "correct", "total", "accuracy_percent"])
        for index, class_name in enumerate(classes):
            correct = int(matrix[index, index])
            total = int(matrix[index].sum())
            accuracy = 100.0 * correct / total if total else 0.0
            writer.writerow([class_name, correct, total, f"{accuracy:.4f}"])


def plot_first_layer_filters(model: nn.Module, output_path: Path) -> None:
    first_conv = next(module for module in model.modules() if isinstance(module, nn.Conv2d))
    filters = first_conv.weight.detach().cpu()
    count = min(filters.shape[0], 64)
    columns = 8
    rows = int(np.ceil(count / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(10, 1.25 * rows))
    axes = np.atleast_1d(axes).ravel()
    for index, axis in enumerate(axes):
        axis.axis("off")
        if index >= count:
            continue
        image = filters[index]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        axis.imshow(image.permute(1, 2, 0).numpy())
    figure.suptitle("First convolutional-layer filters")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_misclassified(
    samples: list[tuple[torch.Tensor, int, int]],
    classes: tuple[str, ...],
    output_path: Path,
) -> None:
    if not samples:
        return
    count = min(len(samples), 20)
    figure, axes = plt.subplots(4, 5, figsize=(12, 9))
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    for axis, (image, target, prediction) in zip(axes.ravel(), samples[:count]):
        image = (image.cpu() * std + mean).clamp(0, 1)
        axis.imshow(image.permute(1, 2, 0).numpy())
        axis.set_title(f"T: {classes[target]}\nP: {classes[prediction]}", fontsize=9)
        axis.axis("off")
    for axis in axes.ravel()[count:]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
