#!/usr/bin/env python
"""06b_severity_sweep — band sweep for the SEVERITY axis.

The cached index only has binary label (=intent). This script maps severity
labels from the source data and runs the same band_sweep logic with
severity >= 1 as the positive class.

    python scripts/06b_severity_sweep.py --config configs/v3_4b.yaml
    python scripts/06b_severity_sweep.py --config configs/v3.yaml
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acsl.eval.invariance import layer_kill_test, multi_axis_band
from acsl.norm import load_norm_stats
from acsl.extract import load_cached
from acsl.runtime import build_parser, resolve, write_manifest


def _discover_layers(cache_dir: str, split: str) -> list[int]:
    split_dir = os.path.join(cache_dir, split)
    if not os.path.isdir(split_dir):
        return []
    layers = []
    for name in os.listdir(split_dir):
        if os.path.isdir(os.path.join(split_dir, name)):
            try:
                layers.append(int(name))
            except ValueError:
                pass
    return sorted(layers)


def _load_severity_map(*data_paths):
    sev = {}
    for p in data_paths:
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                sev[r["id"]] = r.get("severity", 0)
    return sev


def _severity_band_sweep(cache_dir, layers, clean_split, wrapped_split,
                         norm_stats, sev_map, kill_delta_max, auroc_floor):
    """Band sweep using severity labels (>=1 = dangerous)."""
    from sklearn.metrics import roc_auc_score

    per_layer = []
    band = []

    for L in sorted(layers):
        if norm_stats is not None:
            from acsl.norm import load_cached_normalized
            Xc, yc_raw, idc = load_cached_normalized(cache_dir, clean_split, L, norm_stats)
            Xw, yw_raw, idw = load_cached_normalized(cache_dir, wrapped_split, L, norm_stats)
        else:
            Xc, yc_raw, idc = load_cached(cache_dir, clean_split, L)
            Xw, yw_raw, idw = load_cached(cache_dir, wrapped_split, L)

        # Map to severity binary: 0=inert, 1=dangerous (severity >= 1)
        yc_sev = np.array([1 if sev_map.get(str(i), 0) >= 1 else 0 for i in idc])
        yw_sev = np.array([1 if sev_map.get(str(i), 0) >= 1 else 0 for i in idw])

        # Diff-of-means on clean with severity labels
        pos = Xc[yc_sev == 1]
        neg = Xc[yc_sev == 0]
        if pos.shape[0] == 0 or neg.shape[0] == 0:
            per_layer.append({"layer": L, "auroc_clean": float("nan"),
                              "auroc_wrapped": float("nan"), "delta": float("nan"),
                              "in_band": False, "n_pos": 0, "n_neg": 0})
            continue

        diff = pos.mean(axis=0) - neg.mean(axis=0)
        norm = np.linalg.norm(diff)
        if norm < 1e-12:
            per_layer.append({"layer": L, "auroc_clean": float("nan"),
                              "auroc_wrapped": float("nan"), "delta": float("nan"),
                              "in_band": False, "n_pos": int(pos.shape[0]), "n_neg": int(neg.shape[0])})
            continue
        u = (diff / norm).astype(np.float32)

        # AUROC on clean
        scores_c = Xc @ u
        auroc_clean = float(roc_auc_score(yc_sev, scores_c)) if np.unique(yc_sev).size >= 2 else float("nan")

        # AUROC on wrapped
        scores_w = Xw @ u
        auroc_wrapped = float(roc_auc_score(yw_sev, scores_w)) if np.unique(yw_sev).size >= 2 else float("nan")

        delta = float(auroc_clean - auroc_wrapped)
        in_band = (
            not np.isnan(auroc_clean)
            and not np.isnan(auroc_wrapped)
            and auroc_clean >= auroc_floor
            and abs(delta) <= kill_delta_max
        )

        per_layer.append({
            "layer": L,
            "auroc_clean": auroc_clean,
            "auroc_wrapped": auroc_wrapped,
            "delta": delta,
            "in_band": in_band,
            "n_pos": int(pos.shape[0]),
            "n_neg": int(neg.shape[0]),
        })
        if in_band:
            band.append(L)

    return {"per_layer": per_layer, "band": band,
            "kill_delta_max": kill_delta_max, "auroc_floor": auroc_floor}


def main() -> int:
    parser = build_parser(__doc__)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--clean-split", default="clean")
    parser.add_argument("--wrapped-split", default="wrapped")
    parser.add_argument("--no-norm", action="store_true")
    args = parser.parse_args()
    cfg, out = resolve(args)

    cache_dir = args.cache_dir or cfg.get_path("data.cache_dir", os.path.join("runs", "cache"))
    kill_delta_max = float(cfg.get_path("decision.kill_delta_max", 0.05))
    auroc_floor = float(cfg.get_path("decision.auroc_floor", 0.7))

    layers = _discover_layers(cache_dir, args.clean_split)
    if not layers:
        print(f"[severity_sweep] ERROR: no cached layers in {cache_dir}/{args.clean_split}/")
        return 1
    print(f"[severity_sweep] sweeping {len(layers)} layers: {layers[0]}..{layers[-1]}")

    norm_stats = None
    norm_path = os.path.join(cache_dir, "norm_stats.npz")
    if not args.no_norm and os.path.exists(norm_path):
        norm_stats = load_norm_stats(norm_path)
        print(f"[severity_sweep] using normalization from {norm_path}")

    # Load severity labels from source data
    harm_path = cfg.get_path("data.harm_path", "data/v3_harmful.jsonl")
    harmless_path = cfg.get_path("data.harmless_path", "data/v3_harmless.jsonl")
    wrapped_path = cfg.get_path("data.wrapped_path", "data/v3_wrapped.jsonl")
    sev_map = _load_severity_map(harm_path, harmless_path, wrapped_path)
    print(f"[severity_sweep] loaded severity labels for {len(sev_map)} examples")

    # Run severity sweep
    print(f"\n[severity_sweep] === SEVERITY axis (severity >= 1 vs 0) ===")
    sev_sweep = _severity_band_sweep(
        cache_dir, layers, args.clean_split, args.wrapped_split,
        norm_stats, sev_map, kill_delta_max, auroc_floor,
    )

    print(f"{'layer':>6} {'clean':>8} {'wrapped':>8} {'delta':>8} {'band?':>6}")
    print("-" * 42)
    for entry in sev_sweep["per_layer"]:
        mark = " *" if entry["in_band"] else ""
        print(f"{entry['layer']:>6} {entry['auroc_clean']:>8.3f} {entry['auroc_wrapped']:>8.3f} "
              f"{entry['delta']:>+8.3f} {mark:>6}")
    print(f"\n[severity_sweep] severity band B = {sev_sweep['band']}")

    # Also re-run intent for the combined intersection
    from acsl.eval.invariance import band_sweep
    print(f"\n[severity_sweep] === INTENT axis (for intersection) ===")
    intent_sweep = band_sweep(
        cache_dir, layers,
        clean_split=args.clean_split, wrapped_split=args.wrapped_split,
        norm_stats=norm_stats, kill_delta_max=kill_delta_max, auroc_floor=auroc_floor,
    )
    print(f"[severity_sweep] intent band B = {intent_sweep['band']}")

    # Intersection
    axis_sweeps = {"intent": intent_sweep, "severity": sev_sweep}
    result = multi_axis_band(axis_sweeps)

    print(f"\n[severity_sweep] === TWO-AXIS BAND INTERSECTION ===")
    for ax, band in result["per_axis"].items():
        print(f"  {ax}: {band} ({len(band)} layers)")
    print(f"  intersection: {result['band']} ({len(result['band'])} layers)")
    if result["empty_intersection"]:
        print(f"  NOTE: empty intersection — axes invariant at different depths")

    # Save
    output = {
        "layers_swept": layers,
        "kill_delta_max": kill_delta_max,
        "auroc_floor": auroc_floor,
        "normalized": norm_stats is not None,
        "axes": {
            "intent": {"per_layer": intent_sweep["per_layer"], "band": intent_sweep["band"]},
            "severity": {"per_layer": sev_sweep["per_layer"], "band": sev_sweep["band"]},
        },
        "band_intersection": result["band"],
        "empty_intersection": result["empty_intersection"],
    }
    out_path = os.path.join(out, "band_sweep_two_axis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    write_manifest(out, cfg, extra=output)
    print(f"\n[severity_sweep] results -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
