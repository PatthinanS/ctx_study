from __future__ import annotations

from typing import Any, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

_REQUIRED_COLS = {
    "session", "dialog", "utterance_id", "speaker",
    "start_time", "text", "valence", "arousal", "dominance",
}


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def load_iemocap(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"IEMOCAP CSV missing columns: {missing}")
    df = df.dropna(subset=["valence", "arousal", "dominance"]).reset_index(drop=True)

    # Normalise session to int (handles "Ses01" → 1, "Session1" → 1, etc.)
    if df["session"].dtype == object:
        df["session"] = df["session"].str.extract(r"(\d+)").astype(int)

    return df


# ---------------------------------------------------------------------------
# Context strategies
# ---------------------------------------------------------------------------

def _strategy_none(
    utterance: dict,
    history: list[dict],
    k: int,
    **kwargs: Any,
) -> tuple[Optional[str], str]:
    cur_str = f"{utterance['speaker']}: {utterance['text']}"
    return None, cur_str


def _strategy_window(
    utterance: dict,
    history: list[dict],
    k: int,
    **kwargs: Any,
) -> tuple[Optional[str], str]:
    ctx_turns = history[-k:] if k > 0 else []
    ctx_str = " ".join(f"{t['speaker']}: {t['text']}" for t in ctx_turns) if ctx_turns else None
    cur_str = f"{utterance['speaker']}: {utterance['text']}"
    return ctx_str, cur_str


# Registry — add new strategies here; no other file needs to change.
STRATEGY_REGISTRY: dict[str, Any] = {
    "none": _strategy_none,
    "window": _strategy_window,
}


def build_context(
    utterance: dict,
    history: list[dict],
    strategy: str,
    k: int,
    **kwargs: Any,
) -> tuple[Optional[str], str]:
    """Dispatch to the named context strategy.

    Returns (context_str_or_None, current_str).
    Raises ValueError for unknown strategy names.
    """
    if strategy not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown context strategy '{strategy}'. "
            f"Available: {sorted(STRATEGY_REGISTRY)}"
        )
    return STRATEGY_REGISTRY[strategy](utterance, history, k, **kwargs)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class IEMOCAPDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        cfg: dict,
        sessions: list[int],
        max_samples: int | None = None,
    ) -> None:
        ctx_cfg = cfg["context"]
        strategy = ctx_cfg["strategy"]
        k = ctx_cfg["k"]
        strategy_kwargs = ctx_cfg.get("strategy_kwargs", {})
        max_length = cfg["max_length"]

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: list[tuple[Optional[str], str, list[float]]] = []

        subset = df[df["session"].isin(sessions)].copy()
        for (_, _dialog), grp in subset.groupby(["session", "dialog"], sort=False):
            grp = grp.sort_values("start_time").reset_index(drop=True)
            rows = grp.to_dict("records")
            for idx, row in enumerate(rows):
                ctx, cur = build_context(
                    utterance=row,
                    history=rows[:idx],
                    strategy=strategy,
                    k=k,
                    **strategy_kwargs,
                )
                label = [float(row["valence"]), float(row["arousal"]), float(row["dominance"])]
                self.samples.append((ctx, cur, label))
                if max_samples is not None and len(self.samples) >= max_samples:
                    break
            if max_samples is not None and len(self.samples) >= max_samples:
                break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ctx, cur, label = self.samples[idx]

        if ctx is not None:
            enc = self.tokenizer(
                ctx,
                cur,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
        else:
            enc = self.tokenizer(
                cur,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

        out: dict[str, torch.Tensor] = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float32),
        }
        if "token_type_ids" in enc:
            out["token_type_ids"] = enc["token_type_ids"].squeeze(0)
        return out


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def get_splits(df: pd.DataFrame, cfg: dict) -> tuple[list[int], list[int], list[int]]:
    all_sessions = sorted(df["session"].unique().tolist())
    mode = cfg["splits"]["mode"]

    if mode == "leave_one_session_out":
        test = [cfg["splits"]["test_session"]]
        val = [cfg["splits"]["val_session"]]
        train = [s for s in all_sessions if s not in test and s not in val]
    elif mode == "fixed":
        train = cfg["splits"]["train_sessions"]
        val = cfg["splits"]["val_sessions"]
        test = cfg["splits"]["test_sessions"]
    else:
        raise ValueError(f"Unknown split mode '{mode}'")

    for name, lst in [("train", train), ("val", val), ("test", test)]:
        if not lst:
            raise ValueError(f"Empty {name} split after applying split config.")
    return train, val, test


def build_dataloaders(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    cfg: dict,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_sessions, val_sessions, test_sessions = get_splits(df, cfg)

    max_samples = cfg.get("_smoke_max_samples")
    train_ds = IEMOCAPDataset(df, tokenizer, cfg, sessions=train_sessions, max_samples=max_samples)
    val_ds = IEMOCAPDataset(df, tokenizer, cfg, sessions=val_sessions, max_samples=max_samples)
    test_ds = IEMOCAPDataset(df, tokenizer, cfg, sessions=test_sessions, max_samples=max_samples)

    num_workers = cfg.get("num_workers", 4)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader
