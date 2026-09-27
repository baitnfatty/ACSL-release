"""Config loading with attribute + dotted access, and seeding.

A ``Config`` is a plain nested dict that also supports ``cfg.head.hidden`` and
``cfg.get_path("head.hidden", default)``.
"""

from __future__ import annotations

import os
import random


class Config(dict):
    """dict with attribute access; nested dicts are wrapped lazily."""

    def __getattr__(self, key):
        try:
            val = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Config(val) if isinstance(val, dict) else val

    def __setattr__(self, key, value):
        self[key] = value

    def get_path(self, dotted: str, default=None):
        cur = self
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur


def load_config(path: str) -> Config:
    """Load a YAML config file into a Config."""
    import yaml

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping, got {type(data)!r}")
    return Config(data)


def seed_everything(seed: int) -> None:
    """Seed python/numpy/torch (if present) for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover
        pass
