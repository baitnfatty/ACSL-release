"""Difference-of-means caution direction.

This direction is *computed*, never learned (hard constraint 1). At a layer L,
``u = normalize(mean(h_harm) - mean(h_safe))`` is the unit vector along which
harmful-leaning activations sit relative to safe ones. The loop steers along
``-u`` (toward safe) proportionally to the head's risk read.
"""

from __future__ import annotations

import torch


def diff_of_means(h_harm: torch.Tensor, h_safe: torch.Tensor) -> torch.Tensor:
    """Unit vector ``normalize(mean_harm - mean_safe)`` for one layer.

    ``h_harm`` / ``h_safe`` are ``(n_examples, d_model)`` activation matrices.
    Returns a ``(d_model,)`` unit vector (zeros if the means coincide).
    """
    if h_harm.ndim != 2 or h_safe.ndim != 2:
        raise ValueError("expected (n, d_model) matrices")
    if h_harm.shape[1] != h_safe.shape[1]:
        raise ValueError("harm and safe activations must share d_model")

    mu_harm = h_harm.mean(dim=0)
    mu_safe = h_safe.mean(dim=0)
    diff = mu_harm - mu_safe
    norm = torch.linalg.vector_norm(diff)
    if norm <= 0:
        return torch.zeros_like(diff)
    return diff / norm
