"""Post-hoc RMSE evaluation on a trained checkpoint.

Loads the best.pt checkpoint for a given experiment config, runs the test
set through it, and writes per-dimension RMSE to a sibling rmse_log.csv in
the experiment's output_dir (separate from training_log.csv, which holds
Pearson r).

Usage: python eval_rmse.py --config configs/roberta/base/base_roberta.json
"""
from __future__ import annotations

import argparse
import csv
import os

import torch
from transformers import AutoTokenizer

from main import load_config, resolve_device
from src.data import build_dataloaders, load_iemocap
from src.metrics import compute_rmse_metrics, format_rmse_metrics
from src.model import get_model
from src.train import eval_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RMSE on a trained VAD checkpoint")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    parser.add_argument(
        "--device",
        default=None,
        help="Device: cpu | cuda | mps (default: auto-detect)",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Dot-notation config overrides, e.g. --override output_dir=outputs/foo",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.override)
    device = resolve_device(args.device)

    print(f"Device: {device}")
    print(f"Experiment: {cfg.get('experiment_name', args.config)}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["backbone"])
    df = load_iemocap(cfg["csv_path"])

    model = get_model(cfg).to(device)
    output_dir = cfg["output_dir"]
    best_path = os.path.join(output_dir, "best.pt")
    if not os.path.exists(best_path):
        raise FileNotFoundError(
            f"No checkpoint found at {best_path} — train this experiment first (python main.py --config {args.config})"
        )
    model.load_state_dict(torch.load(best_path, map_location=device))

    is_dual = cfg["context"]["strategy"] == "dual_stream"
    _, _, test_loader = build_dataloaders(
        df, tokenizer, cfg,
        backbone=None if is_dual else model.backbone,
        device=device,
    )

    print("\n[Test]")
    preds, labels = eval_epoch(model, test_loader, device)
    metrics = compute_rmse_metrics(preds, labels)
    print(f"  {format_rmse_metrics(metrics)}")

    log_path = os.path.join(output_dir, "rmse_log.csv")
    fieldnames = ["phase", "rmse_valence", "rmse_arousal", "rmse_dominance", "mean_rmse"]
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "phase": "test",
            "rmse_valence": round(metrics["valence"], 6),
            "rmse_arousal": round(metrics["arousal"], 6),
            "rmse_dominance": round(metrics["dominance"], 6),
            "mean_rmse": round(metrics["mean_rmse"], 6),
        })
    print(f"\nRMSE log saved → {log_path}")


if __name__ == "__main__":
    main()
