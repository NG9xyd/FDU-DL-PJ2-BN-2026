from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8") as handle:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def plot_bn_comparison(no_bn_dir: Path, bn_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    no_bn = read_csv(no_bn_dir / "history.csv")
    bn = read_csv(bn_dir / "history.csv")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for rows, label, color in [
        (no_bn, "VGG-A", "#d55e00"),
        (bn, "VGG-A + BN", "#0072b2"),
    ]:
        epochs = [row["epoch"] for row in rows]
        axes[0].plot(epochs, [row["train_loss"] for row in rows], color=color, label=f"{label} train")
        axes[0].plot(
            epochs,
            [row["val_loss"] for row in rows],
            color=color,
            linestyle="--",
            label=f"{label} validation",
        )
        axes[1].plot(
            epochs,
            [row["val_accuracy"] for row in rows],
            color=color,
            label=label,
        )
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Training stability")
    axes[1].set(xlabel="Epoch", ylabel="Validation accuracy (%)", title="Convergence speed")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "bn_training_comparison.png", dpi=200)
    plt.close(figure)


def _smooth(values: np.ndarray, window: int = 25) -> np.ndarray:
    values = values[np.isfinite(values)]
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def create_landscape_plots(results_dir: Path, learning_rates: list[float]) -> None:
    figure, axis = plt.subplots(figsize=(10, 5))
    summary: dict[str, dict[str, float]] = {}
    colors = {"no_bn": "#d55e00", "bn": "#0072b2"}
    landscape_curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for variant, label in [("no_bn", "VGG-A"), ("bn", "VGG-A + BN")]:
        curves: list[np.ndarray] = []
        gradient_changes: list[np.ndarray] = []
        cosine_curves: list[np.ndarray] = []
        for learning_rate in learning_rates:
            rows = read_csv(results_dir / variant / f"lr_{learning_rate:g}" / "steps.csv")
            curves.append(np.array([row["loss"] for row in rows]))
            gradient_changes.append(
                np.array([row["gradient_change_per_distance"] for row in rows])[1:]
            )
            cosine_curves.append(
                np.array([row["gradient_cosine_similarity"] for row in rows])[1:]
            )
        common_length = min(map(len, curves))
        loss_matrix = np.stack([curve[:common_length] for curve in curves])
        minimum = loss_matrix.min(axis=0)
        maximum = loss_matrix.max(axis=0)
        center = loss_matrix.mean(axis=0)
        landscape_curves[variant] = (minimum, maximum, center)
        steps = np.arange(common_length)
        axis.fill_between(
            steps,
            minimum,
            maximum,
            color=colors[variant],
            alpha=0.18,
            label=f"{label} min-max range",
        )
        axis.plot(steps, center, color=colors[variant], linewidth=1.2, label=f"{label} mean")
        flat_changes = np.concatenate(gradient_changes)
        flat_changes = flat_changes[np.isfinite(flat_changes)]
        flat_cosines = np.concatenate(cosine_curves)
        flat_cosines = flat_cosines[np.isfinite(flat_cosines)]
        summary[variant] = {
            "mean_loss_range": float(np.mean(maximum - minimum)),
            "maximum_loss_range": float(np.max(maximum - minimum)),
            "mean_loss_range_after_warmup": float(np.mean((maximum - minimum)[10:])),
            "maximum_loss_range_after_warmup": float(np.max((maximum - minimum)[10:])),
            "median_gradient_change_per_distance": (
                float(np.median(flat_changes)) if len(flat_changes) else float("nan")
            ),
            "mean_gradient_cosine_similarity": (
                float(np.mean(flat_cosines)) if len(flat_cosines) else float("nan")
            ),
        }

    axis.set(
        xlabel="Optimization step",
        ylabel="Training loss",
        title="Loss variation across learning rates",
    )
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(results_dir / "loss_landscape_comparison.png", dpi=200)
    plt.close(figure)

    warmup_steps = 10
    figure, axis = plt.subplots(figsize=(10, 5))
    for variant, label in [("no_bn", "VGG-A"), ("bn", "VGG-A + BN")]:
        minimum, maximum, center = landscape_curves[variant]
        steps = np.arange(len(center))
        axis.fill_between(
            steps[warmup_steps:],
            minimum[warmup_steps:],
            maximum[warmup_steps:],
            color=colors[variant],
            alpha=0.18,
            label=f"{label} min-max range",
        )
        axis.plot(
            steps[warmup_steps:],
            center[warmup_steps:],
            color=colors[variant],
            linewidth=1.2,
            label=f"{label} mean",
        )
    axis.set(
        xlabel="Optimization step",
        ylabel="Training loss",
        title="Loss variation after the first 10 optimization steps",
    )
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(results_dir / "loss_landscape_comparison_zoomed.png", dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for variant, label in [("no_bn", "VGG-A"), ("bn", "VGG-A + BN")]:
        first_lr = learning_rates[0]
        rows = read_csv(results_dir / variant / f"lr_{first_lr:g}" / "steps.csv")
        change = np.array([row["gradient_change_per_distance"] for row in rows])[1:]
        cosine = np.array([row["gradient_cosine_similarity"] for row in rows])[1:]
        axes[0].plot(_smooth(change), color=colors[variant], label=label)
        axes[1].plot(_smooth(cosine), color=colors[variant], label=label)
    axes[0].set(
        xlabel="Optimization step",
        ylabel="Gradient change / parameter distance",
        title="Local gradient variation",
    )
    axes[1].set(
        xlabel="Optimization step",
        ylabel="Cosine similarity",
        title="Consecutive-gradient predictiveness",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(results_dir / "gradient_smoothness_comparison.png", dpi=200)
    plt.close(figure)

    with (results_dir / "landscape_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
