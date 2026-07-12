from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.data import IEMOCAPDataset  # noqa: E402

_BACKBONE = "roberta-base"
_MAX_LENGTH = 128


def _build_dataset(turn0_text: str, turn1_text: str, max_length: int = _MAX_LENGTH) -> IEMOCAPDataset:
    df = pd.DataFrame({
        "session": [1, 1],
        "dialog": ["d1", "d1"],
        "utterance_id": ["u0", "u1"],
        "speaker": ["A", "B"],
        "start_time": [0.0, 1.0],
        "text": [turn0_text, turn1_text],
        "valence": [3.0, 3.0],
        "arousal": [3.0, 3.0],
        "dominance": [3.0, 3.0],
    })
    cfg = {
        "backbone": _BACKBONE,
        "max_length": max_length,
        "context": {"strategy": "window", "k": 1, "strategy_kwargs": {}},
    }
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(_BACKBONE)
        return IEMOCAPDataset(df, tokenizer, cfg, sessions=[1])
    except OSError as e:
        pytest.skip(f"could not load backbone {_BACKBONE!r}: {e}")


def _text_with_token_count(tokenizer, n_tokens: int) -> str:
    words = ["the"] * (n_tokens + 20)
    text = " ".join(words)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
    result = tokenizer.decode(ids, skip_special_tokens=True)
    check_ids = tokenizer(result, add_special_tokens=False)["input_ids"]
    assert len(check_ids) == n_tokens, f"expected {n_tokens} tokens, got {len(check_ids)}"
    return result


def test_target_never_truncated_ctx_drops_oldest_tokens() -> None:
    ds = _build_dataset(
        turn0_text=_text_with_token_count(_dummy_tokenizer(), 200),
        turn1_text=_text_with_token_count(_dummy_tokenizer(), 40),
    )
    ctx, cur, _ = ds.samples[1]
    assert ctx is not None
    assert not ds._cur_exceeds_budget(cur)

    item = ds[1]
    input_ids = item["input_ids"].tolist()
    real_len = int(item["attention_mask"].sum().item())
    real_ids = input_ids[:real_len]

    cur_ids = ds.tokenizer(cur, add_special_tokens=False)["input_ids"]
    full_ctx_ids = ds.tokenizer(ctx, add_special_tokens=False)["input_ids"]

    # cur must appear intact, as a contiguous block right before the final EOS.
    assert real_ids[-(len(cur_ids) + 1):-1] == cur_ids

    # ctx occupies everything between the leading BOS and the separator run
    # before cur; it must be a *suffix* (most recent tokens) of the full ctx,
    # not a prefix.
    mid_sep_count = ds._pair_overhead - 2
    cur_start = len(real_ids) - 1 - len(cur_ids)
    ctx_end = cur_start - mid_sep_count
    ctx_survived_ids = real_ids[1:ctx_end]
    assert 0 < len(ctx_survived_ids) < len(full_ctx_ids)
    assert ctx_survived_ids == full_ctx_ids[-len(ctx_survived_ids):]


def test_cur_far_over_budget_falls_back_without_error() -> None:
    ds = _build_dataset(
        turn0_text=_text_with_token_count(_dummy_tokenizer(), 50),
        turn1_text=_text_with_token_count(_dummy_tokenizer(), _MAX_LENGTH + 50),
    )
    ctx, cur, _ = ds.samples[1]
    assert ctx is not None
    assert ds._cur_exceeds_budget(cur)

    item = ds[1]  # must not raise
    assert item["input_ids"].shape[0] == _MAX_LENGTH


def test_cur_at_exact_boundary_triggers_fallback() -> None:
    tokenizer = _dummy_tokenizer()
    overhead = tokenizer.num_special_tokens_to_add(pair=True)
    boundary_cur = _text_with_token_count(tokenizer, _MAX_LENGTH - overhead)

    ds = _build_dataset(
        turn0_text=_text_with_token_count(tokenizer, 50),
        turn1_text=boundary_cur,
    )
    ctx, cur, _ = ds.samples[1]
    assert ds._cur_exceeds_budget(cur)  # exactly at boundary must still fall back

    item = ds[1]
    expected = ds.tokenizer(
        cur, max_length=_MAX_LENGTH, truncation=True, padding="max_length", return_tensors="pt",
    )
    assert item["input_ids"].tolist() == expected["input_ids"].squeeze(0).tolist()


_tok_cache: dict[str, object] = {}


def _dummy_tokenizer():
    if "tok" not in _tok_cache:
        try:
            from transformers import AutoTokenizer
            _tok_cache["tok"] = AutoTokenizer.from_pretrained(_BACKBONE)
        except OSError as e:
            pytest.skip(f"could not load backbone {_BACKBONE!r}: {e}")
    return _tok_cache["tok"]
