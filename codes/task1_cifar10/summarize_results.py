from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine experiment summaries into one CSV.")
    parser.add_argument("--results-dir", type=Path, default=Path("results/task1"))
    parser.add_argument("--output", type=Path, default=Path("results/task1_summary.csv"))
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for summary_path in sorted(args.results_dir.glob("*/summary.json")):
        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        config_path = summary_path.with_name("config.json")
        with config_path.open(encoding="utf-8") as handle:
            config = json.load(handle)
        rows.append(
            {
                "experiment": summary["experiment_name"],
                "channels": ",".join(str(value) for value in config["channels"]),
                "activation": config["activation"],
                "batch_norm": not config["no_batch_norm"],
                "dropout": config["dropout"],
                "loss": config["loss"],
                "label_smoothing": config["label_smoothing"],
                "optimizer": config["optimizer"],
                "learning_rate": config["lr"],
                "weight_decay": config["weight_decay"],
                "parameters": summary["parameters"],
                "best_epoch": summary["best_epoch"],
                "best_val_accuracy": summary["best_val_accuracy"],
                "test_accuracy": summary["test_accuracy"],
                "test_error": summary["test_error"],
                "training_seconds": summary["training_seconds"],
            }
        )

    if not rows:
        raise SystemExit(f"No summary.json files found below {args.results_dir}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} experiments to {args.output}")


if __name__ == "__main__":
    main()
