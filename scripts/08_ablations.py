#!/usr/bin/env python
"""08_ablations - what actually carries the effect?

  random-direction control : loop separation (benign-safe minus harm-safe) using
                             the diff-of-means u vs a random unit direction.
  K=1 vs K>1               : does iterating the loop help? safe fractions per K.
  single vs multi-layer    : head AUROC on the best single layer vs features
                             concatenated across the cached layers.
  probe input note         : probe-only (o_L) vs raw-attention (h_L) requires a
                             second cache built with `01_extract --which h_L`.

Reads the activation cache (01) and a trained head (02).

    python scripts/08_ablations.py --config configs/default.yaml --head runs/02_sweep_train_head/head.pt
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from acsl.directions import diff_of_means
from acsl.extract import load_cached
from acsl.head import fit_head, load_head
from acsl.loop import security_loop
from acsl.runtime import build_parser, resolve, write_manifest


def _safe_frac(feats, head, u, params):
    if len(feats) == 0:
        return float("nan")
    return float(np.mean([
        1.0 if security_loop(torch.as_tensor(x, dtype=torch.float32), head, u, **params)[2] else 0.0
        for x in feats
    ]))


def _split(n, seed):
    perm = np.random.default_rng(seed).permutation(n)
    cut = max(1, int(0.8 * n))
    va = perm[cut:]
    return perm[:cut], (va if len(va) else perm[:cut])


def main() -> int:
    parser = build_parser(__doc__)
    parser.add_argument("--head", default=os.path.join("runs", "02_sweep_train_head", "head.pt"))
    args = parser.parse_args()
    cfg, out = resolve(args)

    seed = int(cfg.get_path("seed", 0))
    head, meta = load_head(args.head)
    layer = meta.get("layer")
    cache_dir = cfg.get_path("data.cache_dir", os.path.join("runs", "cache"))

    Xc, yc, idc = load_cached(cache_dir, "clean", layer)
    Xt = torch.as_tensor(Xc, dtype=torch.float32)
    u = diff_of_means(Xt[yc == 1], Xt[yc == 0])
    harm_feats, benign_feats = Xc[yc == 1], Xc[yc == 0]

    g_max = float(cfg.get_path("loop.g_max", 8.0))
    eps = float(cfg.get_path("loop.eps", 1e-2))
    thr = float(cfg.get_path("loop.thr", 0.0))
    K = int(cfg.get_path("loop.K", 4))

    report = {"layer": layer}

    # 1) random-direction control --------------------------------------------
    rng = np.random.default_rng(seed)
    rand = torch.as_tensor(rng.normal(size=u.shape), dtype=torch.float32)
    rand = rand / rand.norm().clamp(min=1e-12)
    params = dict(g_max=g_max, K=K, eps=eps, thr=thr)
    gap_u = _safe_frac(benign_feats, head, u, params) - _safe_frac(harm_feats, head, u, params)
    gap_rand = _safe_frac(benign_feats, head, rand, params) - _safe_frac(harm_feats, head, rand, params)
    report["random_direction_control"] = {"gap_u": gap_u, "gap_random": gap_rand}

    # 2) K=1 vs K>1 -----------------------------------------------------------
    k_sweep = {}
    for k in sorted({1, K}):
        p = dict(g_max=g_max, K=k, eps=eps, thr=thr)
        k_sweep[k] = {
            "benign_safe": _safe_frac(benign_feats, head, u, p),
            "harm_safe": _safe_frac(harm_feats, head, u, p),
        }
    report["K_sweep"] = k_sweep

    # 3) single vs multi-layer ------------------------------------------------
    n_categories = int(cfg.get_path("head.n_categories", 2))
    hidden = cfg.get_path("head.hidden", None)
    split_dir = os.path.join(cache_dir, "clean")
    layers_present = sorted(int(d) for d in os.listdir(split_dir) if d.isdigit()) if os.path.isdir(split_dir) else []

    def _auroc_on(X, y):
        tr, va = _split(len(y), seed)
        _, auroc = fit_head(X[tr], y[tr], X[va], y[va], n_categories=n_categories, hidden=hidden,
                            epochs=int(cfg.get_path("head.epochs", 20)), seed=seed)
        return auroc

    single = _auroc_on(Xc, yc)
    multi = None
    if len(layers_present) > 1:
        per_layer = {}            # layer -> {id: feature vector}
        for L in layers_present:
            XL, _yL, idL = load_cached(cache_dir, "clean", L)
            per_layer[L] = {i: v for i, v in zip(idL, XL)}
        labels = {i: int(l) for i, l in zip(idc, yc)}
        common = set.intersection(*[set(per_layer[L]) for L in layers_present])
        common = [i for i in idc if i in common]
        if common:
            Xcat = np.stack([np.concatenate([per_layer[L][i] for L in layers_present]) for i in common])
            ycat = np.asarray([labels[i] for i in common], np.int64)
            multi = _auroc_on(Xcat, ycat)
    report["single_vs_multi_layer"] = {
        "single_layer_auroc": single, "multi_layer_auroc": multi, "layers": layers_present,
    }

    # 4) probe input note -----------------------------------------------------
    report["probe_input_note"] = (
        "probe-only (o_L) vs raw-attention (h_L): build a second cache with "
        "`01_extract --which h_L` and re-run 02/this script to compare."
    )

    print(json.dumps(report, indent=2))
    with open(os.path.join(out, "ablations.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    write_manifest(out, cfg, extra=report)
    print(f"[ablations] OK - manifest -> {os.path.join(out, 'manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
