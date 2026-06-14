from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data import build_loaders
from experiment import set_seed, train_model
from models import VGGA, count_parameters
from plots import create_landscape_plots, plot_bn_comparison


def default_data_dir() -> Path:
    candidates = [
        Path("../task1_cifar10/dataset"),
        Path("codes/task1_cifar10/dataset"),
        Path("../../codes/task1_cifar10/dataset"),
    ]
    for candidate in candidates:
        if (candidate / "cifar-10-batches-py").exists():
            return candidate
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VGG-A batch-normalization experiments.")
    parser.add_argument("--mode", choices=["compare", "landscape", "all"], default="all")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--landscape-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--learning-rates", default="0.0001,0.0005,0.001,0.002")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--train-items", type=int, default=-1)
    parser.add_argument("--landscape-items", type=int, default=10000)
    parser.add_argument("--test-items", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--fake-data", action="store_true")
    parser.add_argument("--max-steps", type=int, default=-1)
    return parser.parse_args()


def run_pair(args: argparse.Namespace, device: torch.device) -> None:
    compare_dir = args.output_dir / "comparison"
    summaries = {}
    for use_bn, name in [(False, "no_bn"), (True, "bn")]:
        set_seed(args.seed)
        loaders = build_loaders(
            args.data_dir,
            args.batch_size,
            args.num_workers,
            args.val_size,
            args.seed,
            train_items=args.train_items,
            test_items=args.test_items,
            augment=args.augment,
            fake_data=args.fake_data,
        )
        model = VGGA(batch_norm=use_bn)
        print(f"\n{name}: {count_parameters(model):,} parameters")
        result = train_model(
            model=model,
            loaders=loaders,
            device=device,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            optimizer_name=args.optimizer,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            output_dir=compare_dir / name,
            use_amp=not args.no_amp,
            max_steps=args.max_steps,
        )
        summaries[name] = {
            "best_val_accuracy": result.best_val_accuracy,
            "test_accuracy": result.test_accuracy,
            "best_epoch": result.best_epoch,
            "elapsed_seconds": result.elapsed_seconds,
        }
    plot_bn_comparison(compare_dir / "no_bn", compare_dir / "bn", compare_dir)
    with (compare_dir / "comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)


def run_landscape(args: argparse.Namespace, device: torch.device) -> None:
    learning_rates = [float(value) for value in args.learning_rates.split(",")]
    landscape_dir = args.output_dir / "landscape"
    for use_bn, name in [(False, "no_bn"), (True, "bn")]:
        for learning_rate in learning_rates:
            set_seed(args.seed)
            loaders = build_loaders(
                args.data_dir,
                args.batch_size,
                args.num_workers,
                args.val_size,
                args.seed,
                train_items=args.landscape_items,
                test_items=args.test_items,
                augment=False,
                fake_data=args.fake_data,
            )
            print(f"\n{name}, learning rate={learning_rate:g}")
            train_model(
                model=VGGA(batch_norm=use_bn),
                loaders=loaders,
                device=device,
                epochs=args.landscape_epochs,
                learning_rate=learning_rate,
                optimizer_name=args.optimizer,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                output_dir=landscape_dir / name / f"lr_{learning_rate:g}",
                # Gradient diagnostics need unscaled FP32 gradients.
                use_amp=False,
                max_steps=args.max_steps,
            )
    create_landscape_plots(landscape_dir, learning_rates)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, default=str)
    if args.mode in {"compare", "all"}:
        run_pair(args, device)
    if args.mode in {"landscape", "all"}:
        run_landscape(args, device)


if __name__ == "__main__":
    main()
