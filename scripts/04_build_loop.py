#!/usr/bin/env python
"""04_build_loop - compute u at L, assemble the loop, calibrate its thresholds.

Computes the diff-of-means caution direction u at the chosen layer, then scans
the risk threshold thr (with g_max/K/eps from config) on a dev split to maximize
the gap between 'benign resolves safe' and 'harmful does NOT resolve safe'. Saves
u and the calibrated loop params for the eval.

    python scripts/04_build_loop.py --config configs/default.yaml --head runs/02_sweep_train_head/head.pt
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
from acsl.head import load_head
from acsl.loop import security_loop
from acsl.runtime import build_parser, resolve, write_manifest


def _safe_fraction(feats, head, u, params) -> float:
    if len(feats) == 0:
        return float("nan")
    flags = []
    for x in feats:
        _, _, safe = security_loop(torch.as_tensor(x, dtype=torch.float32), head, u, **params)
        flags.append(1.0 if safe else 0.0)
    return float(np.mean(flags))


def main() -> int:
    parser = build_parser(__doc__)
    parser.add_argument("--head", default=os.path.join("runs", "02_sweep_train_head", "head.pt"))
    args = parser.parse_args()
    cfg, out = resolve(args)

    head, meta = load_head(args.head)
    layer = meta.get("layer")
    cache_dir = cfg.get_path("data.cache_dir", os.path.join("runs", "cache"))

    Xc, yc, _ = load_cached(cache_dir, "clean", layer)
    if Xc.shape[0] == 0:
        raise FileNotFoundError(f"no 'clean' features at layer {layer}; run 01/02 first")
    Xt = torch.as_tensor(Xc, dtype=torch.float32)
    u = diff_of_means(Xt[yc == 1], Xt[yc == 0])

    harm_feats = Xc[yc == 1]
    benign_split_dir = os.path.join(cache_dir, "benign")
    if os.path.isdir(benign_split_dir):
        Xb, yb, _ = load_cached(cache_dir, "benign", layer)
        benign_feats = Xb if Xb.shape[0] else Xc[yc == 0]
    else:
        benign_feats = Xc[yc == 0]

    g_max = float(cfg.get_path("loop.g_max", 8.0))
    K = int(cfg.get_path("loop.K", 4))
    eps = float(cfg.get_path("loop.eps", 1e-2))

    # Calibrate thr: maximize (benign safe) - (harm safe).
    best = None
    grid = np.linspace(-4.0, 4.0, 33)
    sweep = []
    for thr in grid:
        params = dict(g_max=g_max, K=K, eps=eps, thr=float(thr))
        benign_safe = _safe_fraction(benign_feats, head, u, params)
        harm_safe = _safe_fraction(harm_feats, head, u, params)
        gap = (0.0 if benign_safe != benign_safe else benign_safe) - (0.0 if harm_safe != harm_safe else harm_safe)
        sweep.append({"thr": float(thr), "benign_safe": benign_safe, "harm_safe": harm_safe, "gap": gap})
        if best is None or gap > best["gap"]:
            best = sweep[-1]

    params = {"g_max": g_max, "K": K, "eps": eps, "thr": best["thr"], "layer": layer}
    np.save(os.path.join(out, "u.npy"), u.cpu().numpy())
    with open(os.path.join(out, "loop_params.json"), "w", encoding="utf-8") as f:
        json.dump({"params": params, "calibration": best, "sweep": sweep}, f, indent=2)

    print(f"[loop] layer={layer} calibrated thr={best['thr']:.3f} "
          f"(benign_safe={best['benign_safe']:.2f} harm_safe={best['harm_safe']:.2f})")
    write_manifest(out, cfg, extra={"params": params, "calibration": best})
    print(f"[loop] OK - u + params saved under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
