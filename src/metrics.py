from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr

_DIMS = ("valence", "arousal", "dominance")


def pearson_per_dim(
    preds: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """Pearson r for each of the three VAD dimensions.

    Returns {"valence": r, "arousal": r, "dominance": r}.
    Returns nan for any dimension where preds or labels are constant
    (avoids raising inside scipy on degenerate batches).
    """
    result = {}
    for i, dim in enumerate(_DIMS):
        p, l = preds[:, i], labels[:, i]
        if np.std(p) < 1e-8 or np.std(l) < 1e-8:
            result[dim] = float("nan")
        else:
            result[dim] = float(pearsonr(p, l)[0])
    return result


def compute_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """Pearson r per dim plus mean_pearson (ignoring nans)."""
    metrics = pearson_per_dim(preds, labels)
    vals = [v for v in metrics.values() if not np.isnan(v)]
    metrics["mean_pearson"] = float(np.mean(vals)) if vals else float("nan")
    return metrics


def format_metrics(metrics: dict[str, float]) -> str:
    parts = [f"r_{dim[0].upper()}={metrics[dim]:.4f}" for dim in _DIMS]
    parts.append(f"mean_r={metrics['mean_pearson']:.4f}")
    return "  ".join(parts)
