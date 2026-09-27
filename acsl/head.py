"""SecurityHead: a small classifier read off cached activations.

The base transformer is frozen; only this head is trained. ``risk_logit`` turns
the head's per-category logits into a single log-odds-of-harm scalar (centered
at 0, so ``loop.thr=0.0`` means p(harm)=0.5). The loop and policy both read this
internal scalar — never output tokens.
"""

from __future__ import annotations

import os

import torch
from torch import nn


class SecurityHead(nn.Module):
    """Linear probe (``hidden=None``) or 1-hidden-layer MLP over a d_model vector."""

    def __init__(self, d_model: int, n_categories: int, hidden: int | None = None):
        super().__init__()
        self.d_model = int(d_model)
        self.n_categories = int(n_categories)
        self.hidden = hidden
        if hidden:
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Linear(hidden, n_categories),
            )
        else:
            self.net = nn.Linear(d_model, n_categories)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (..., n_categories) risk logits


def risk_logit(logits: torch.Tensor, benign_index: int = 0, temperature: float = 1.0) -> torch.Tensor:
    """Collapse per-category logits to a scalar harm log-odds.

    For 3 categories (safe=0, intent=1, dangerous=2):
      ``r = logit[2] - logit[0]`` — dangerous vs safe, intent is the learned middle.

    For 2 categories:
      ``r = logsumexp(non-benign) - benign`` ≈ log p(harm)/p(benign).

    For a single-logit head, the lone logit is the risk directly.

    ``temperature`` > 1 softens the output (spreads sigmoid away from 0/1).
    """
    c = logits.shape[-1]
    if c == 1:
        r = logits[..., 0]
    elif c == 3:
        r = logits[..., 2] - logits[..., benign_index]
    else:
        benign = logits[..., benign_index]
        keep = [i for i in range(c) if i != benign_index]
        nonbenign = logits[..., keep]
        r = torch.logsumexp(nonbenign, dim=-1) - benign
    if temperature != 1.0:
        r = r / temperature
    return r


def _binary_target(y: torch.Tensor, benign_index: int = 0) -> torch.Tensor:
    """harm(1) vs benign(0) target for AUROC."""
    return (y != benign_index).long()


def _stratified_split(y, frac_train: float = 0.8, rng=None):
    """Class-stratified train/val index split so both classes reach val.

    A class with a single example is placed in both train and val.
    """
    import numpy as np

    rng = rng or np.random.default_rng(0)
    y = np.asarray(y)
    tr, va = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        if len(idx) == 1:
            tr.append(idx); va.append(idx)
            continue
        cut = max(1, int(round(frac_train * len(idx))))
        cut = min(cut, len(idx) - 1)  # leave at least one for val
        tr.append(idx[:cut]); va.append(idx[cut:])
    tr = np.concatenate(tr) if tr else np.array([], int)
    va = np.concatenate(va) if va else np.array([], int)
    return tr, va


def save_head(head: SecurityHead, path: str, layer: int | None = None, val_auroc: float | None = None) -> None:
    """Serialize a SecurityHead (state + architecture) for later loading."""
    torch.save(
        {
            "state_dict": head.state_dict(),
            "d_model": head.d_model,
            "n_categories": head.n_categories,
            "hidden": head.hidden,
            "layer": layer,
            "val_auroc": val_auroc,
        },
        path,
    )


def load_head(path: str) -> tuple[SecurityHead, dict]:
    """Reconstruct a SecurityHead from ``save_head``; returns ``(head, meta)``."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    head = SecurityHead(ckpt["d_model"], ckpt["n_categories"], hidden=ckpt["hidden"])
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return head, meta


def train_head(acts_dir, labels, layer, cfg) -> tuple[SecurityHead, float]:
    """Train a SecurityHead on cached activations; return ``(head, val_auroc)``.

    ``acts_dir`` is the activation cache root, ``layer`` the tapped layer. Splits
    named ``train``/``val`` (or ``dev``) are used if present; otherwise the single
    available split is divided 80/20. ``labels`` optionally overrides the cached
    per-id labels (dict id->int). ``cfg`` provides ``head.*``, ``seed``. The base
    model is never touched — we only read ``.npy`` features from disk.
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score

    from .config import Config
    from .extract import load_cached

    cfg = cfg if isinstance(cfg, Config) else Config(cfg if isinstance(cfg, dict) else {})
    seed = int(cfg.get_path("seed", 0))
    hidden = cfg.get_path("head.hidden", None)
    n_categories = int(cfg.get_path("head.n_categories", 2))
    lr = float(cfg.get_path("head.lr", 1e-3))
    epochs = int(cfg.get_path("head.epochs", 20))
    benign_index = int(cfg.get_path("head.benign_index", 0))

    rng = np.random.default_rng(seed)

    def _maybe_override(ids, y):
        if labels is None:
            return y
        return np.asarray([int(labels.get(str(i), int(yy))) for i, yy in zip(ids, y)], np.int64)

    # Discover splits present under acts_dir.
    splits = [
        d for d in (os.listdir(acts_dir) if os.path.isdir(acts_dir) else [])
        if os.path.isdir(os.path.join(acts_dir, d))
    ]
    # Prefer an explicit/canonical training split over an arbitrary one (a
    # single-class split like 'benign' would make AUROC undefined).
    preferred = cfg.get_path("data.train_split", None)
    train_candidates = ([preferred] if preferred else []) + ["train", "clean"]
    train_split = next((s for s in train_candidates if s in splits), None)
    val_split = next((s for s in ("val", "dev", "validation") if s in splits), None)
    if train_split is None and splits:
        train_split = splits[0]

    if train_split and val_split:
        Xtr, ytr, idtr = load_cached(acts_dir, train_split, layer)
        Xva, yva, idva = load_cached(acts_dir, val_split, layer)
        ytr = _maybe_override(idtr, ytr)
        yva = _maybe_override(idva, yva)
    else:
        one = train_split or (splits[0] if splits else None)
        if one is None:
            raise FileNotFoundError(f"no cached splits found under {acts_dir!r}")
        X, y, ids = load_cached(acts_dir, one, layer)
        y = _maybe_override(ids, y)
        tr_idx, va_idx = _stratified_split(y, frac_train=0.8, rng=rng)
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xva, yva = X[va_idx], y[va_idx]

    if Xtr.shape[0] == 0:
        raise ValueError("no training activations loaded; run extraction first")

    return fit_head(
        Xtr, ytr, Xva, yva,
        n_categories=n_categories, hidden=hidden, lr=lr, epochs=epochs,
        seed=seed, benign_index=benign_index,
    )


def fit_head(
    X_train, y_train, X_val, y_val,
    n_categories: int, hidden: int | None = None,
    lr: float = 1e-3, epochs: int = 20, seed: int = 0, benign_index: int = 0,
) -> tuple[SecurityHead, float]:
    """Train a SecurityHead on in-memory arrays; return ``(head, val_auroc)``.

    Pure head training — no base model involved. Used by ``train_head`` and by
    the ablations script (e.g. multi-layer concatenated features).
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(seed)
    X_train = np.asarray(X_train, np.float32)
    X_val = np.asarray(X_val, np.float32)
    y_train = np.asarray(y_train, np.int64)
    y_val = np.asarray(y_val, np.int64)

    head = SecurityHead(X_train.shape[1], n_categories, hidden=hidden)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    xt = torch.as_tensor(X_train)
    yt = torch.as_tensor(y_train).clamp(0, n_categories - 1)

    head.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(head(xt), yt)
        loss.backward()  # gradients flow ONLY through the head, never the base
        opt.step()

    head.eval()
    with torch.no_grad():
        scores = risk_logit(head(torch.as_tensor(X_val)), benign_index=benign_index).numpy()
    y_bin = (y_val != benign_index).astype(int)
    val_auroc = float("nan") if np.unique(y_bin).size < 2 else float(roc_auc_score(y_bin, scores))
    return head, val_auroc
