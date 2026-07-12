from __future__ import annotations

import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.data import build_dataloaders, load_iemocap
from src.metrics import compute_metrics, format_metrics
from src.model import get_loss_fn, get_model


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Optimizer / scheduler
# ---------------------------------------------------------------------------

def build_optimizer(
    model: VADModel,
    phase_cfg: dict,
    freeze_encoder: bool,
) -> AdamW:
    if freeze_encoder:
        optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=phase_cfg["lr"])
    else:
        # Differential LR: smaller rate for backbone, larger for everything else.
        multiplier = phase_cfg.get("head_lr_multiplier", 10)
        backbone_ids = {id(p) for p in model.backbone.parameters()}
        # Non-backbone modules (head, and fusion for dual-stream) are randomly
        # initialized, so they all share the head's elevated LR.
        rest = [p for p in model.parameters() if id(p) not in backbone_ids]
        optimizer = AdamW([
            {"params": list(model.backbone.parameters()), "lr": phase_cfg["lr"]},
            {"params": rest, "lr": phase_cfg["lr"] * multiplier},
        ])

    registered = {id(p) for g in optimizer.param_groups for p in g["params"]}
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert trainable <= registered, "optimizer is missing trainable parameters"
    return optimizer


def build_scheduler(
    optimizer: AdamW,
    n_training_steps: int,
    warmup_ratio: float,
) -> object:
    warmup_steps = int(n_training_steps * warmup_ratio)
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=n_training_steps,
    )


# ---------------------------------------------------------------------------
# Single epoch helpers
# ---------------------------------------------------------------------------

def _get_batch_kwargs(batch: dict, device: torch.device) -> dict:
    kwargs: dict = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "token_type_ids": batch["token_type_ids"].to(device) if "token_type_ids" in batch else None,
    }
    if "same_input_ids" in batch:
        for k in ("same_input_ids", "same_attention_mask", "same_valid",
                  "cross_input_ids", "cross_attention_mask", "cross_valid"):
            kwargs[k] = batch[k].to(device)
    return kwargs


def train_epoch(
    model: VADModel,
    loader: DataLoader,
    optimizer: AdamW,
    scheduler: object,
    loss_fn: nn.Module,
    cfg: dict,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    clip = cfg.get("gradient_clip", 1.0)

    for batch in tqdm(loader, desc="  train", leave=False):
        kwargs = _get_batch_kwargs(batch, device)
        labels = batch["labels"].to(device)

        preds = model(**kwargs)
        loss = loss_fn(preds, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(
    model: VADModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="  eval", leave=False):
        kwargs = _get_batch_kwargs(batch, device)
        preds = model(**kwargs).cpu()
        all_preds.append(preds)
        all_labels.append(batch["labels"])

    return (
        torch.cat(all_preds).numpy(),
        torch.cat(all_labels).numpy(),
    )


# ---------------------------------------------------------------------------
# Phase runner
# ---------------------------------------------------------------------------

def run_phase(
    model: VADModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    cfg: dict,
    device: torch.device,
    phase_cfg: dict,
    phase_name: str,
    output_dir: str,
    log_rows: list[dict],
) -> VADModel:
    n_epochs = phase_cfg["epochs"]
    n_steps = n_epochs * len(train_loader)
    optimizer = build_optimizer(model, phase_cfg, freeze_encoder=phase_cfg.get("freeze_encoder", False))
    scheduler = build_scheduler(optimizer, n_steps, phase_cfg.get("warmup_ratio", 0.1))

    best_path = os.path.join(output_dir, "best.pt")
    last_path = os.path.join(output_dir, "last.pt")
    best_r = float("-inf")

    for ep in range(1, n_epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, cfg, device)
        preds, labels = eval_epoch(model, val_loader, device)
        metrics = compute_metrics(preds, labels)

        line = f"  [{phase_name}] ep {ep:>2}/{n_epochs}  loss={tr_loss:.4f}  {format_metrics(metrics)}"
        print(line)

        # Always save last checkpoint
        torch.save(model.state_dict(), last_path)

        # Save best checkpoint when val Pearson improves
        if metrics["mean_pearson"] > best_r:
            best_r = metrics["mean_pearson"]
            torch.save(model.state_dict(), best_path)

        log_rows.append({
            "phase": phase_name,
            "epoch": ep,
            "train_loss": round(tr_loss, 6),
            "r_valence": round(metrics["valence"], 6),
            "r_arousal": round(metrics["arousal"], 6),
            "r_dominance": round(metrics["dominance"], 6),
            "mean_pearson": round(metrics["mean_pearson"], 6),
            "best_so_far": round(best_r, 6),
        })

    model.load_state_dict(torch.load(best_path, map_location=device))
    return model


# ---------------------------------------------------------------------------
# Full training pipeline
# ---------------------------------------------------------------------------

def train(cfg: dict, device: torch.device) -> dict[str, float]:
    set_seed(cfg.get("seed", 42))

    print(f"Loading tokenizer: {cfg['backbone']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["backbone"])

    print(f"Loading IEMOCAP from: {cfg['csv_path']}")
    df = load_iemocap(cfg["csv_path"])
    print(f"  {len(df)} utterances, {df['session'].nunique()} sessions")

    model = get_model(cfg).to(device)
    loss_fn = get_loss_fn(cfg).to(device)

    is_dual = cfg["context"]["strategy"] == "dual_stream"
    train_loader, val_loader, test_loader = build_dataloaders(
        df, tokenizer, cfg,
        backbone=None if is_dual else model.backbone,
        device=device,
    )
    print(
        f"  train={len(train_loader.dataset)}  "
        f"val={len(val_loader.dataset)}  "
        f"test={len(test_loader.dataset)}"
    )

    output_dir = cfg.get("output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    log_rows: list[dict] = []

    # Phase 1: encoder frozen
    print("\n[Phase 1] Encoder frozen — training head only")
    model.freeze_encoder()
    model = run_phase(
        model, train_loader, val_loader, loss_fn, cfg, device,
        phase_cfg=cfg["phase1"], phase_name="phase1",
        output_dir=output_dir, log_rows=log_rows,
    )

    # Phase 2: full fine-tune
    print("\n[Phase 2] All parameters unfrozen")
    model.unfreeze_encoder()
    model = run_phase(
        model, train_loader, val_loader, loss_fn, cfg, device,
        phase_cfg=cfg["phase2"], phase_name="phase2",
        output_dir=output_dir, log_rows=log_rows,
    )

    # Final test evaluation
    print("\n[Test]")
    preds, labels = eval_epoch(model, test_loader, device)
    metrics = compute_metrics(preds, labels)
    print(f"  {format_metrics(metrics)}")

    log_rows.append({
        "phase": "test",
        "epoch": "-",
        "train_loss": "-",
        "r_valence": round(metrics["valence"], 6),
        "r_arousal": round(metrics["arousal"], 6),
        "r_dominance": round(metrics["dominance"], 6),
        "mean_pearson": round(metrics["mean_pearson"], 6),
        "best_so_far": "-",
    })

    # Write CSV log
    log_path = os.path.join(output_dir, "training_log.csv")
    fieldnames = ["phase", "epoch", "train_loss", "r_valence", "r_arousal", "r_dominance",
                  "mean_pearson", "best_so_far"]
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"\nLog saved → {log_path}")

    return metrics
