from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.model import VADModel, VADModelDualStream  # noqa: E402


def _discover_backbones() -> list[str]:
    backbones = set()
    for path in (_REPO_ROOT / "configs").rglob("*.json"):
        cfg = json.loads(path.read_text())
        if "backbone" in cfg:
            backbones.add(cfg["backbone"])
    return sorted(backbones)


@pytest.mark.parametrize("backbone", _discover_backbones())
def test_vad_model_has_no_pooler(backbone: str) -> None:
    try:
        model = VADModel({"backbone": backbone, "dropout": 0.1})
    except OSError as e:
        pytest.skip(f"could not load backbone {backbone!r}: {e}")
    assert model.backbone.pooler is None


@pytest.mark.parametrize("backbone", _discover_backbones())
def test_vad_model_dual_stream_has_no_pooler(backbone: str) -> None:
    try:
        model = VADModelDualStream({"backbone": backbone, "dropout": 0.1})
    except OSError as e:
        pytest.skip(f"could not load backbone {backbone!r}: {e}")
    assert model.backbone.pooler is None
