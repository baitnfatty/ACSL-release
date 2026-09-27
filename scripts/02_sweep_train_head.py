#!/usr/bin/env python
"""02_sweep_train_head - train a SecurityHead per layer; pick the best layer by
   kill-test INVARIANCE, not raw AUROC.

Reads cached activations (01_extract), trains the head on the 'clean' split at
each layer, and (if a 'wrapped' split exists) also measures the head's AUROC on
jailbreak-wrapped prompts. Selection: the most *discriminative* layer (highest
clean AUROC) **among layers whose clean-vs-wrapped delta <= decision.kill_delta_max**
(i.e. the signal survives wrapping). This avoids picking a high-AUROC layer that
is surface-form dependent. Falls back to clean AUROC when no wrapped split exists.

    python scripts/02_sweep_train_head.py --config configs/default.yaml
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from sklearn.metrics import roc_auc_score

from acsl.extract import load_cached
from acsl.head import risk_logit, save_head, train_head
from acsl.runtime import build_parser, layer_range, resolve, write_manifest


def main() -> int:
    parser = build_parser(__doc__)
    parser.add_argument("--split", default="clean", help="cache split to train on")
    parser.add_argument("--layer", type=int, default=None, help="train on a single layer (skip sweep)")
    args = parser.parse_args()
    cfg, out = resolve(args)
    cfg.setdefault("data", {})["train_split"] = args.split  # train on this split

    cache_dir = cfg.get_path("data.cache_dir", os.path.join("runs", "cache"))
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(f"no activation cache at {cache_dir!r}; run 01_extract first")

    # Only sweep layers that actually have cached features for the split.
    present = set()
    split_dir = os.path.join(cache_dir, args.split)
    if os.path.isdir(split_dir):
        present = {int(d) for d in os.listdir(split_dir) if d.isdigit()}
    if args.layer is not None:
        if args.layer not in present:
            raise FileNotFoundError(f"layer {args.layer} not cached under {split_dir!r}")
        layers = [args.layer]
    else:
        layers = [L for L in layer_range(cfg) if L in present] or sorted(present)
    if not layers:
        raise FileNotFoundError(f"no cached layers under {split_dir!r}")

    kill_delta_max = float(cfg.get_path("decision.kill_delta_max", 0.05))
    benign_index = int(cfg.get_path("head.benign_index", 0))
    wrapped_present = os.path.isdir(os.path.join(cache_dir, "wrapped"))

    def _nan_safe(x):
        return -1.0 if (x != x) else x

    results, heads = [], {}
    for L in layers:
        head, clean_auroc = train_head(cache_dir, None, L, cfg)
        heads[L] = head
        row = {"layer": L, "clean_auroc": clean_auroc}
        if wrapped_present:
            Xw, yw, _ = load_cached(cache_dir, "wrapped", L)
            yb = (yw != benign_index).astype(int)
            if len(yb) and np.unique(yb).size > 1:
                with torch.no_grad():
                    rw = risk_logit(head(torch.as_tensor(Xw, dtype=torch.float32)),
                                    benign_index=benign_index).numpy()
                row["wrapped_auroc"] = float(roc_auc_score(yb, rw))
                row["delta"] = clean_auroc - row["wrapped_auroc"]
        results.append(row)
        msg = f"[sweep] layer {L:>2}  clean AUROC={clean_auroc:.4f}"
        if "delta" in row:
            inv = "  invariant" if abs(row["delta"]) <= kill_delta_max else "  KILLED"
            msg += f"  wrapped={row['wrapped_auroc']:.4f}  delta={row['delta']:+.4f}{inv}"
        print(msg)

    # --- selection: invariance-gated discriminativeness --------------------
    deltas = [r for r in results if "delta" in r]
    if deltas:
        invariant = [r for r in deltas if abs(r["delta"]) <= kill_delta_max]
        if invariant:
            best_row = max(invariant, key=lambda r: _nan_safe(r["clean_auroc"]))
            criterion = f"most discriminative among invariant (delta<={kill_delta_max})"
        else:
            best_row = min(deltas, key=lambda r: r["delta"])
            criterion = "no layer met the invariance threshold; chose most invariant"
    else:
        best_row = max(results, key=lambda r: _nan_safe(r["clean_auroc"]))
        criterion = "no wrapped split; chose highest clean AUROC"

    best_layer = best_row["layer"]
    best_head = heads[best_layer]
    best_auroc = best_row["clean_auroc"]
    head_path = os.path.join(out, "head.pt")
    save_head(best_head, head_path, layer=best_layer, val_auroc=best_auroc)
    with open(os.path.join(out, "sweep.json"), "w", encoding="utf-8") as f:
        json.dump({"results": results, "best_layer": best_layer, "criterion": criterion}, f, indent=2)

    print(f"[sweep] selection: {criterion}")
    print(f"[sweep] best layer = {best_layer}  clean AUROC={best_auroc:.4f}"
          + (f"  delta={best_row['delta']:+.4f}" if "delta" in best_row else "")
          + f"  -> {head_path}")
    write_manifest(out, cfg, extra={"best_layer": best_layer, "criterion": criterion, "results": results})
    print(f"[sweep] OK - manifest -> {os.path.join(out, 'manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
