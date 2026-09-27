"""Per-layer normalization for residual-stream activations.

Residual-stream norms grow with depth.  Without correction, raw projections
at layer 4 and layer 28 are on different scales, and trajectory deltas
partly measure norm growth instead of signal change.

Fix: z-score per layer using training-set statistics.  Compute mu and sigma
on the training split ONCE, then apply identically to every split (train,
eval, held-out).  Never fit normalization on eval data.

Persist mu/sigma in the run manifest so eval uses the same transform.
"""

from __future__ import annotations

import json
import os

import numpy as np


def compute_norm_stats(cache_dir: str, train_split: str, layers: list[int]) -> dict:
    """Compute per-layer mean and std from cached training-split activations.

    Returns ``{layer: {"mu": ndarray[d_model], "sigma": ndarray[d_model]}}``
    with sigma floored at 1e-6 to avoid division by zero.
    """
    from .extract import load_cached

    stats = {}
    for L in layers:
        X, _, _ = load_cached(cache_dir, train_split, L)
        if X.shape[0] == 0:
            raise ValueError(f"no cached activations for split={train_split!r} layer={L}")
        mu = X.mean(axis=0).astype(np.float32)
        sigma = X.std(axis=0).astype(np.float32)
        sigma = np.maximum(sigma, 1e-6)
        stats[L] = {"mu": mu, "sigma": sigma}
    return stats


def save_norm_stats(stats: dict, path: str) -> None:
    """Save normalization stats to an npz file.

    Keys are ``mu_{layer}`` and ``sigma_{layer}`` for each layer.
    Also saves a ``layers`` array for discovery.
    """
    arrays = {"layers": np.array(sorted(stats.keys()), dtype=np.int32)}
    for L, s in stats.items():
        arrays[f"mu_{L}"] = s["mu"]
        arrays[f"sigma_{L}"] = s["sigma"]
    np.savez(path, **arrays)


def load_norm_stats(path: str) -> dict:
    """Load normalization stats from an npz file saved by ``save_norm_stats``."""
    data = np.load(path)
    layers = data["layers"].tolist()
    stats = {}
    for L in layers:
        stats[L] = {
            "mu": data[f"mu_{L}"].astype(np.float32),
            "sigma": data[f"sigma_{L}"].astype(np.float32),
        }
    return stats


def normalize(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Z-score normalize: ``(X - mu) / sigma``.

    X is (n_examples, d_model); mu and sigma are (d_model,).
    """
    return ((X - mu) / sigma).astype(np.float32)


def load_cached_normalized(cache_dir: str, split: str, layer: int,
                           norm_stats: dict) -> tuple[np.ndarray, np.ndarray, list]:
    """Load cached activations and apply per-layer z-score normalization.

    ``norm_stats`` is the dict from ``compute_norm_stats`` or ``load_norm_stats``.
    Returns (X_normalized, y, ids).
    """
    from .extract import load_cached

    X, y, ids = load_cached(cache_dir, split, layer)
    if X.shape[0] == 0:
        return X, y, ids
    if layer not in norm_stats:
        raise KeyError(f"no normalization stats for layer {layer}")
    s = norm_stats[layer]
    X_norm = normalize(X, s["mu"], s["sigma"])
    return X_norm, y, ids
