"""Entry point: python main.py --config configs/base_roberta.json"""
from __future__ import annotations

import argparse
import json
import sys

import torch

from src.train import train


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _set_nested(d: dict, key_path: str, value: str) -> None:
    """Set a dot-notation key path in a nested dict. Values are JSON-parsed."""
    keys = key_path.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    try:
        d[keys[-1]] = json.loads(value)
    except json.JSONDecodeError:
        d[keys[-1]] = value  # treat as string if not valid JSON


def load_config(path: str, overrides: list[str]) -> dict:
    with open(path) as f:
        cfg = json.load(f)
    for override in overrides:
        if "=" not in override:
            print(f"Warning: ignoring malformed override '{override}' (expected key=value)", file=sys.stderr)
            continue
        key, _, value = override.partition("=")
        _set_nested(cfg, key.strip(), value.strip())
    return cfg


# ---------------------------------------------------------------------------
# Smoke-test injection: limits data and epochs for fast CPU sanity checks
# ---------------------------------------------------------------------------

def _apply_smoke(cfg: dict) -> dict:
    cfg["phase1"]["epochs"] = 1
    cfg["phase2"]["epochs"] = 1
    cfg["num_workers"] = 0
    cfg["batch_size"] = 8
    cfg["_smoke_max_samples"] = 64
    return cfg


# ---------------------------------------------------------------------------
# C1 vs C2 sanity check
# ---------------------------------------------------------------------------

def _sanity_check(cfg: dict, device: torch.device) -> None:
    from transformers import AutoModel, AutoTokenizer

    from src.data import _embed_dialogue, build_context, load_iemocap

    print(f"[sanity] Loading tokenizer and backbone: {cfg['backbone']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["backbone"])
    df = load_iemocap(cfg["csv_path"])
    backbone = AutoModel.from_pretrained(cfg["backbone"]).to(device).eval()

    n_shown = 0
    for (_session, _dialog), grp in df.groupby(["session", "dialog"], sort=False):
        if n_shown >= 3:
            break
        rows = grp.sort_values("start_time").to_dict("records")
        if len(rows) < 4:
            continue

        embeddings = _embed_dialogue(rows, backbone, tokenizer, device, cfg.get("max_length", 128))

        print(f"\n--- Dialogue: {_dialog} (session {_session}) ---")
        for idx in range(3, min(6, len(rows))):
            row, history = rows[idx], rows[:idx]
            ctx_cfg = cfg["context"]
            ctx_c1, _ = build_context(row, history, strategy="window", k=ctx_cfg["k"])
            ctx_c2, _ = build_context(
                row, history, strategy="retrieval", k=ctx_cfg["k"],
                embeddings=embeddings, turn_idx=idx,
                **ctx_cfg.get("strategy_kwargs", {}),
            )
            print(f"  utt[{idx}] target : {row['speaker']}: {row['text']}")
            print(f"           C1 ctx  : {ctx_c1}")
            print(f"           C2 ctx  : {ctx_c2}")
            print()
        n_shown += 1

    print("[sanity] Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VAD regression on IEMOCAP")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Limit to 1 epoch per phase and 0 workers — fast CPU sanity check",
    )
    parser.add_argument(
        "--sanity",
        action="store_true",
        help="Print C1 vs C2 context comparison on 3 sample dialogues, then exit",
    )
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
        help="Dot-notation config overrides, e.g. --override loss.name=emd_batch batch_size=8",
    )
    return parser.parse_args()


def resolve_device(requested: str | None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.override)

    if args.smoke:
        cfg = _apply_smoke(cfg)
        print("[smoke mode] 1 epoch per phase, num_workers=0")

    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Experiment: {cfg.get('experiment_name', args.config)}")

    if args.sanity:
        _sanity_check(cfg, device)
        return

    metrics = train(cfg, device)

    print("\n=== Final test metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
