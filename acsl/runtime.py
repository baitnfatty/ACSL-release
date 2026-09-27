"""Run-time helpers: reproducible manifests + a common CLI scaffold.

Every ``scripts/0X_*.py`` builds its parser with ``build_parser`` (so each gets
``--config`` and ``--help``) and calls ``write_manifest`` to record the resolved
config, git SHA, and a content hash of the eval sets it touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from typing import Iterable

from .config import Config, load_config, seed_everything

DEFAULT_CONFIG = os.path.join("configs", "default.yaml")


def git_sha() -> str:
    """Short git SHA of the working tree, or 'nogit' when unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip() or "nogit"
    except Exception:
        pass
    return "nogit"


def content_hash(paths: Iterable[str | None]) -> dict:
    """sha256 of each existing path (file contents). Missing/None -> 'absent'."""
    digests: dict[str, str] = {}
    for p in paths:
        if not p:
            continue
        if not os.path.exists(p):
            digests[p] = "absent"
            continue
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        digests[p] = h.hexdigest()
    return digests


def build_parser(description: str) -> argparse.ArgumentParser:
    """Common parser: every script gets --config, --out, --seed, --help."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=DEFAULT_CONFIG, help="path to YAML config")
    p.add_argument("--out", default=None, help="output/run directory (default: runs/<script>)")
    p.add_argument("--seed", type=int, default=None, help="override config seed")
    return p


def resolve(args) -> tuple[Config, str]:
    """Load config, apply seed override, seed RNGs, ensure out dir exists."""
    cfg = load_config(args.config)
    if getattr(args, "seed", None) is not None:
        cfg["seed"] = args.seed
    seed_everything(int(cfg.get("seed", 0)))

    script = os.path.splitext(os.path.basename(sys.argv[0]))[0] or "run"
    out = args.out or os.path.join("runs", script)
    os.makedirs(out, exist_ok=True)
    return cfg, out


def load_model(cfg: Config, fake: bool = False):
    """Load the (frozen) base model, real or offline-fake. Returns (model, tokenizer)."""
    if fake:
        from ._fake import load_fake_model

        return load_fake_model(d_model=64, n_layers=6)
    from .model import load_frozen_model

    return load_frozen_model(cfg.get_path("model.name"), cfg.get_path("model.dtype", "float16"))


def build_examples(cfg: Config, fake: bool = False, limit=None):
    """Examples across eval conditions: synthetic when fake, else from cfg.data.*."""
    if fake:
        from .data.synthetic_smoke import synthetic_conditions

        return synthetic_conditions(n_per_group=8)
    from .data.loaders import condition_examples

    return list(
        condition_examples(
            harm_path=cfg.get_path("data.harm_path"),
            harmless_path=cfg.get_path("data.harmless_path"),
            wrapped_path=cfg.get_path("data.wrapped_path"),
            benign_sensitive_path=cfg.get_path("data.benign_sensitive_path"),
            limit=limit,
            harmless_wrapped_only=bool(cfg.get_path("data.harmless_wrapped_only", False)),
        )
    )


def layer_range(cfg: Config) -> list[int]:
    """Inclusive layer range from cfg.tap.layer_range -> list of ints."""
    lo, hi = cfg.get_path("tap.layer_range", [0, 0])
    return list(range(int(lo), int(hi) + 1))


def write_manifest(out_dir: str, cfg: Config, eval_paths: Iterable[str | None] = (), extra: dict | None = None) -> str:
    """Write ``manifest.json`` capturing config + git SHA + eval-set hashes.

    Note: time.time() is used only to stamp the manifest, never inside model code.
    """
    manifest = {
        "timestamp": time.time(),
        "argv": sys.argv,
        "git_sha": git_sha(),
        "config": dict(cfg),
        "eval_set_hashes": content_hash(eval_paths),
    }
    if extra:
        manifest["extra"] = extra
    path = os.path.join(out_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    return path
