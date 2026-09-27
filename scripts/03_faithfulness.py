#!/usr/bin/env python
"""03_faithfulness - is the head's signal grounded, internal, and non-spurious?

  grounding   : perturbing h along the caution direction u moves the head's risk
                (vs a random-direction control that should not).
  internality : risk correlates with the activation's projection onto u (the
                head reads the internal state, not surface tokens).
  spuriousness: risk should NOT be explained by trivial magnitude features like
                the activation L2 norm.

Reads the activation cache (01) and a trained head (02).

    python scripts/03_faithfulness.py --config configs/default.yaml --head runs/02_sweep_train_head/head.pt
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
from acsl.head import load_head, risk_logit
from acsl.runtime import build_parser, resolve, write_manifest


def _risk(head, X) -> np.ndarray:
    with torch.no_grad():
        return risk_logit(head(torch.as_tensor(np.asarray(X), dtype=torch.float32))).cpu().numpy().ravel()


def _corr(a, b) -> float:
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main() -> int:
    parser = build_parser(__doc__)
    parser.add_argument("--head", default=os.path.join("runs", "02_sweep_train_head", "head.pt"))
    parser.add_argument("--split", default="clean")
    args = parser.parse_args()
    cfg, out = resolve(args)

    head, meta = load_head(args.head)
    layer = meta.get("layer")
    cache_dir = cfg.get_path("data.cache_dir", os.path.join("runs", "cache"))
    X, y, _ = load_cached(cache_dir, args.split, layer)
    if X.shape[0] == 0:
        raise FileNotFoundError(f"no cached features for split={args.split!r} layer={layer}")

    Xt = torch.as_tensor(X, dtype=torch.float32)
    u = diff_of_means(Xt[y == 1], Xt[y == 0])
    u_np = u.cpu().numpy()
    rng = np.random.default_rng(int(cfg.get_path("seed", 0)))
    rand = rng.normal(size=u_np.shape)
    rand = rand / (np.linalg.norm(rand) + 1e-12)

    # --- grounding: risk vs perturbation magnitude along u and a random dir ----
    ts = np.linspace(-3.0, 3.0, 13)
    sample = X[: min(64, X.shape[0])]
    corr_u, corr_rand = [], []
    for x in sample:
        risks_u = _risk(head, x[None, :] + ts[:, None] * u_np[None, :])
        risks_r = _risk(head, x[None, :] + ts[:, None] * rand[None, :])
        corr_u.append(_corr(ts, risks_u))
        corr_rand.append(_corr(ts, risks_r))
    grounding_u = float(np.mean(corr_u))
    grounding_rand = float(np.mean(corr_rand))

    # --- internality & spuriousness -------------------------------------------
    risk_all = _risk(head, X)
    proj_u = X @ u_np
    norms = np.linalg.norm(X, axis=1)
    internality = _corr(risk_all, proj_u)
    spurious_norm = _corr(risk_all, norms)

    report = {
        "layer": layer,
        "grounding_corr_u": grounding_u,
        "grounding_corr_random_control": grounding_rand,
        "internality_corr_risk_vs_proj_u": internality,
        "spuriousness_corr_risk_vs_norm": spurious_norm,
        "interpretation": {
            "grounded": abs(grounding_u) > 0.5 and abs(grounding_u) > 3 * abs(grounding_rand),
            "magnitude_spurious_flag": abs(spurious_norm) > 0.5,
        },
    }
    print(json.dumps(report, indent=2))
    with open(os.path.join(out, "faithfulness.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    write_manifest(out, cfg, extra=report)
    print(f"[faithfulness] OK - manifest -> {os.path.join(out, 'manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
