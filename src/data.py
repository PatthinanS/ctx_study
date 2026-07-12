from __future__ import annotations

import warnings
from typing import Any, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

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
# Embedding helper (used by retrieval strategy)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _embed_dialogue(
    rows: list[dict],
    backbone: Any,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    max_length: int = 128,
) -> torch.Tensor:
    """Mean-pool last hidden state over non-padding tokens for each turn."""
    texts = [f"{r['speaker']}: {r['text']}" for r in rows]
    enc = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to(device)
    outputs = backbone(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    mask = enc["attention_mask"].unsqueeze(-1).float()
    emb = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return emb.cpu()


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


_asc_deprecation_warned = False


def _strategy_retrieval(
    utterance: dict,
    history: list[dict],
    k: int,
    embeddings: "torch.Tensor | None" = None,
    turn_idx: int = 0,
    sort_by: str = "chrono",
    **kwargs: Any,
) -> tuple[Optional[str], str]:
    cur_str = f"{utterance['speaker']}: {utterance['text']}"
    if not history or embeddings is None:
        return None, cur_str
    if sort_by == "asc":
        global _asc_deprecation_warned
        if not _asc_deprecation_warned:
            msg = "sort_by='asc' is deprecated; use 'sim_asc' instead."
            # DeprecationWarning is ignored by default outside __main__ (PEP 565),
            # so also print — otherwise this would silently never surface when
            # run via main.py/run_experiments.sh.
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            print(f"  [deprecation] {msg}")
            _asc_deprecation_warned = True
        sort_by = "sim_asc"
    target_emb = embeddings[turn_idx].unsqueeze(0)   # (1, H)
    history_embs = embeddings[:turn_idx]              # (n_prior, H)
    sims = F.cosine_similarity(target_emb, history_embs, dim=-1)
    topk = min(k, len(history))
    topk_result = sims.topk(topk)
    if sort_by == "sim_asc":
        # ascending similarity: least similar first, most similar last (sits closest to target in input)
        order = topk_result.values.argsort()
        top_indices = topk_result.indices[order].tolist()
    else:  # "chrono": original dialogue order
        top_indices = topk_result.indices.sort().values.tolist()
    ctx_str = " ".join(f"{t['speaker']}: {t['text']}" for t in [history[i] for i in top_indices])
    return ctx_str, cur_str


# Registry — add new strategies here; no other file needs to change.
STRATEGY_REGISTRY: dict[str, Any] = {
    "none": _strategy_none,
    "window": _strategy_window,
    "retrieval": _strategy_retrieval,
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
# Truncation diagnostics (construction-time only, print-only)
# ---------------------------------------------------------------------------

def _print_truncation_stats(label: str, attn_sums: list[int], max_length: int) -> None:
    n = len(attn_sums)
    if n == 0:
        print(f"  [truncation] {label}: no samples")
        return
    arr = torch.tensor(attn_sums, dtype=torch.float32)
    trunc_rate = (arr == max_length).float().mean().item()
    mean_tokens = arr.mean().item()
    print(f"  [truncation] {label}: n={n} truncation_rate={trunc_rate:.3f} mean_non_pad_tokens={mean_tokens:.1f}")


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
        backbone: Any = None,
        device: "torch.device | None" = None,
    ) -> None:
        ctx_cfg = cfg["context"]
        strategy = ctx_cfg["strategy"]
        k = ctx_cfg["k"]
        strategy_kwargs = ctx_cfg.get("strategy_kwargs", {})
        max_length = cfg["max_length"]

        self.tokenizer = tokenizer
        self.max_length = max_length
        # Dedicated left-truncating tokenizer so pair encoding can drop ctx's
        # oldest tokens first without mutating the shared self.tokenizer
        # (which stays right-truncating, used for cur and the ctx-is-None path).
        self._left_tokenizer = AutoTokenizer.from_pretrained(cfg["backbone"], truncation_side="left")
        self._pair_overhead = self.tokenizer.num_special_tokens_to_add(pair=True)
        self.samples: list[tuple[Optional[str], str, list[float]]] = []

        subset = df[df["session"].isin(sessions)].copy()
        for (_, _dialog), grp in subset.groupby(["session", "dialog"], sort=False):
            grp = grp.sort_values("start_time").reset_index(drop=True)
            rows = grp.to_dict("records")

            dial_embeddings = None
            if strategy == "retrieval" and backbone is not None:
                backbone.eval()
                dial_embeddings = _embed_dialogue(
                    rows, backbone, tokenizer,
                    device or torch.device("cpu"), max_length,
                )

            for idx, row in enumerate(rows):
                ctx, cur = build_context(
                    utterance=row,
                    history=rows[:idx],
                    strategy=strategy,
                    k=k,
                    embeddings=dial_embeddings,
                    turn_idx=idx,
                    **strategy_kwargs,
                )
                label = [float(row["valence"]), float(row["arousal"]), float(row["dominance"])]
                self.samples.append((ctx, cur, label))
                if max_samples is not None and len(self.samples) >= max_samples:
                    break
            if max_samples is not None and len(self.samples) >= max_samples:
                break

        self._print_truncation_stats(sessions)

    def _cur_exceeds_budget(self, cur: str) -> bool:
        # Exact boundary (not an approximation): HF's only_first truncation
        # requires ctx to retain >=1 token after truncation, which reduces to
        # cur_len + overhead < max_length for success. only_first does NOT
        # raise when this fails — it just logs an error and returns the
        # untruncated ids — so this proactive check is the real safety
        # mechanism; a try/except around the only_first call would catch
        # nothing.
        cur_len = len(self.tokenizer(cur, add_special_tokens=False)["input_ids"])
        return cur_len + self._pair_overhead >= self.max_length

    def _print_truncation_stats(self, sessions: list[int]) -> None:
        none_ctx_curs = [cur for ctx, cur, _ in self.samples if ctx is None]
        pair_samples = [(ctx, cur) for ctx, cur, _ in self.samples if ctx is not None]
        exceeds = [self._cur_exceeds_budget(cur) for _, cur in pair_samples]
        normal_pairs = [p for p, e in zip(pair_samples, exceeds) if not e]
        fallback_pairs = [p for p, e in zip(pair_samples, exceeds) if e]

        attn_sums: list[int] = []
        if none_ctx_curs:
            enc = self.tokenizer(
                none_ctx_curs, max_length=self.max_length, truncation=True,
                padding=True, return_tensors="pt",
            )
            attn_sums.extend(enc["attention_mask"].sum(dim=1).tolist())
        if normal_pairs:
            ctxs, curs = zip(*normal_pairs)
            enc = self._left_tokenizer(
                list(ctxs), list(curs), max_length=self.max_length, truncation="only_first",
                padding=True, return_tensors="pt",
            )
            attn_sums.extend(enc["attention_mask"].sum(dim=1).tolist())
        if fallback_pairs:
            _, curs = zip(*fallback_pairs)
            enc = self.tokenizer(
                list(curs), max_length=self.max_length, truncation=True,
                padding=True, return_tensors="pt",
            )
            attn_sums.extend(enc["attention_mask"].sum(dim=1).tolist())

        _print_truncation_stats(f"IEMOCAPDataset sessions={sessions}", attn_sums, self.max_length)
        print(f"  [truncation] IEMOCAPDataset sessions={sessions}: cur_exceeds_budget_fallback_count={len(fallback_pairs)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ctx, cur, label = self.samples[idx]

        if ctx is not None:
            if self._cur_exceeds_budget(cur):
                enc = self.tokenizer(
                    cur,
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
            else:
                enc = self._left_tokenizer(
                    ctx,
                    cur,
                    max_length=self.max_length,
                    truncation="only_first",
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
# C3: Speaker-stratified dual-stream dataset
# ---------------------------------------------------------------------------

class IEMOCAPDualStreamDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        cfg: dict,
        sessions: list[int],
        max_samples: int | None = None,
    ) -> None:
        max_length = cfg["max_length"]
        stream_k = cfg["context"].get("stream_k", 0)
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Same/cross streams are truncated from the left (oldest turns dropped
        # first) so long dialogues keep the most recent, contextually relevant
        # turns. Target-utterance tokenization is untouched (default right side).
        self._left_tokenizer = AutoTokenizer.from_pretrained(cfg["backbone"], truncation_side="left")
        self.samples: list[tuple[Optional[str], Optional[str], str, list[float]]] = []

        subset = df[df["session"].isin(sessions)].copy()
        for (_, _dialog), grp in subset.groupby(["session", "dialog"], sort=False):
            grp = grp.sort_values("start_time").reset_index(drop=True)
            rows = grp.to_dict("records")

            for idx, row in enumerate(rows):
                history = rows[:idx]
                spk = row["speaker"]

                same_turns = [t for t in history if t["speaker"] == spk]
                cross_turns = [t for t in history if t["speaker"] != spk]
                if stream_k > 0:
                    same_turns = same_turns[-stream_k:]
                    cross_turns = cross_turns[-stream_k:]

                same_text  = " ".join(f"{t['speaker']}: {t['text']}" for t in same_turns) or None
                cross_text = " ".join(f"{t['speaker']}: {t['text']}" for t in cross_turns) or None
                cur_text   = f"{spk}: {row['text']}"

                label = [float(row["valence"]), float(row["arousal"]), float(row["dominance"])]
                self.samples.append((same_text, cross_text, cur_text, label))

                if max_samples is not None and len(self.samples) >= max_samples:
                    break
            if max_samples is not None and len(self.samples) >= max_samples:
                break

        self._print_truncation_stats(sessions)

    def _print_truncation_stats(self, sessions: list[int]) -> None:
        same_texts  = [s if s is not None else " " for s, _, _, _ in self.samples]
        cross_texts = [c if c is not None else " " for _, c, _, _ in self.samples]

        same_enc = self._left_tokenizer(
            same_texts, max_length=self.max_length, truncation=True,
            padding=True, return_tensors="pt",
        )
        cross_enc = self._left_tokenizer(
            cross_texts, max_length=self.max_length, truncation=True,
            padding=True, return_tensors="pt",
        )
        _print_truncation_stats(
            f"IEMOCAPDualStreamDataset same sessions={sessions}",
            same_enc["attention_mask"].sum(dim=1).tolist(), self.max_length,
        )
        _print_truncation_stats(
            f"IEMOCAPDualStreamDataset cross sessions={sessions}",
            cross_enc["attention_mask"].sum(dim=1).tolist(), self.max_length,
        )

    def _tokenize(
        self, text: Optional[str], tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> dict[str, torch.Tensor]:
        tok = tokenizer or self.tokenizer
        src = text if text is not None else " "
        enc = tok(
            src,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        same_text, cross_text, cur_text, label = self.samples[idx]

        target_enc = self._tokenize(cur_text)
        same_enc   = self._tokenize(same_text, self._left_tokenizer)
        cross_enc  = self._tokenize(cross_text, self._left_tokenizer)

        return {
            "input_ids":            target_enc["input_ids"],
            "attention_mask":       target_enc["attention_mask"],
            "same_input_ids":       same_enc["input_ids"],
            "same_attention_mask":  same_enc["attention_mask"],
            "same_valid":           torch.tensor(same_text is not None, dtype=torch.bool),
            "cross_input_ids":      cross_enc["input_ids"],
            "cross_attention_mask": cross_enc["attention_mask"],
            "cross_valid":          torch.tensor(cross_text is not None, dtype=torch.bool),
            "labels":               torch.tensor(label, dtype=torch.float32),
        }


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
    backbone: Any = None,
    device: "torch.device | None" = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_sessions, val_sessions, test_sessions = get_splits(df, cfg)

    max_samples = cfg.get("_smoke_max_samples")
    if cfg["context"]["strategy"] == "dual_stream":
        train_ds = IEMOCAPDualStreamDataset(df, tokenizer, cfg, sessions=train_sessions, max_samples=max_samples)
        val_ds   = IEMOCAPDualStreamDataset(df, tokenizer, cfg, sessions=val_sessions,   max_samples=max_samples)
        test_ds  = IEMOCAPDualStreamDataset(df, tokenizer, cfg, sessions=test_sessions,  max_samples=max_samples)
    else:
        train_ds = IEMOCAPDataset(df, tokenizer, cfg, sessions=train_sessions, max_samples=max_samples, backbone=backbone, device=device)
        val_ds   = IEMOCAPDataset(df, tokenizer, cfg, sessions=val_sessions,   max_samples=max_samples, backbone=backbone, device=device)
        test_ds  = IEMOCAPDataset(df, tokenizer, cfg, sessions=test_sessions,  max_samples=max_samples, backbone=backbone, device=device)

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
