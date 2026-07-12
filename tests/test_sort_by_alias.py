from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import src.data as data  # noqa: E402


def _make_history_and_embeddings():
    history = [
        {"speaker": "A", "text": "hello there"},
        {"speaker": "B", "text": "how are you"},
        {"speaker": "A", "text": "good and you"},
    ]
    utterance = {"speaker": "B", "text": "doing fine thanks"}
    embeddings = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5],
        [0.9, 0.1],  # target embedding (turn_idx=3)
    ])
    return utterance, history, embeddings


def test_sim_asc_matches_legacy_asc_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data, "_asc_deprecation_warned", False)
    utterance, history, embeddings = _make_history_and_embeddings()

    with pytest.warns(DeprecationWarning):
        legacy = data._strategy_retrieval(
            utterance, history, k=2, embeddings=embeddings, turn_idx=3, sort_by="asc",
        )

    monkeypatch.setattr(data, "_asc_deprecation_warned", False)
    renamed = data._strategy_retrieval(
        utterance, history, k=2, embeddings=embeddings, turn_idx=3, sort_by="sim_asc",
    )

    assert legacy == renamed


def test_asc_warns_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data, "_asc_deprecation_warned", False)
    utterance, history, embeddings = _make_history_and_embeddings()

    with pytest.warns(DeprecationWarning) as record:
        data._strategy_retrieval(utterance, history, k=2, embeddings=embeddings, turn_idx=3, sort_by="asc")
        data._strategy_retrieval(utterance, history, k=2, embeddings=embeddings, turn_idx=3, sort_by="asc")
        data._strategy_retrieval(utterance, history, k=2, embeddings=embeddings, turn_idx=3, sort_by="asc")

    deprecation_warnings = [w for w in record.list if issubclass(w.category, DeprecationWarning)]
    assert len(deprecation_warnings) == 1
