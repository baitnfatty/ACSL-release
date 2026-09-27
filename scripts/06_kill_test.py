#!/usr/bin/env python
"""06_kill_test - per-layer band sweep: find invariant layers for each axis.

v3: sweeps ALL cached layers with diff-of-means projections (no trained head
needed).  For each axis with labels, computes AUROC on clean and wrapped splits
at every layer.  Reports the invariant band B per axis and the intersection.

An empty intersection is a finding (axes are invariant at different depths),
not a bug to fix by loosening thresholds.

    python scripts/06_kill_test.py --config configs/v3.yaml
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acsl.eval.invariance import band_sweep, multi_axis_band
from acsl.norm import load_norm_stats
from acsl.runtime import build_parser, resolve, write_manifest


def _discover_layers(cache_dir: str, split: str) -> list[int]:
    """Find which layers have cached activations for a split."""
    split_dir = os.path.join(cache_dir, split)
    if not os.path.isdir(split_dir):
        return []
    layers = []
    for name in os.listdir(split_dir):
        d = os.path.join(split_dir, name)
        if os.path.isdir(d):
            try:
                layers.append(int(name))
            except ValueError:
                pass
    return sorted(layers)


def main() -> int:
    parser = build_parser(__doc__)
    parser.add_argument("--cache-dir", default=None,
                        help="activation cache root (default: from config data.cache_dir)")
    parser.add_argument("--clean-split", default="clean")
    parser.add_argument("--wrapped-split", default="wrapped")
    parser.add_argument("--no-norm", action="store_true",
                        help="skip per-layer normalization (not recommended)")
    args = parser.parse_args()
    cfg, out = resolve(args)

    cache_dir = args.cache_dir or cfg.get_path("data.cache_dir", os.path.join("runs", "cache"))
    kill_delta_max = float(cfg.get_path("decision.kill_delta_max", 0.05))
    auroc_floor = float(cfg.get_path("decision.auroc_floor", 0.7))

    # Discover available layers from cache
    layers = _discover_layers(cache_dir, args.clean_split)
    if not layers:
        print(f"[kill_test] ERROR: no cached layers found in {cache_dir}/{args.clean_split}/")
        return 1
    print(f"[kill_test] sweeping {len(layers)} layers: {layers[0]}..{layers[-1]}")

    # Load normalization stats
    norm_stats = None
    norm_path = os.path.join(cache_dir, "norm_stats.npz")
    if not args.no_norm and os.path.exists(norm_path):
        norm_stats = load_norm_stats(norm_path)
        print(f"[kill_test] using per-layer normalization from {norm_path}")
    elif not args.no_norm:
        print(f"[kill_test] WARNING: no norm_stats.npz found at {norm_path}, running without normalization")

    # --- Intent axis (label: 0=benign, >=1=harmful) ---
    print(f"\n[kill_test] === INTENT axis (label >= 1 vs 0) ===")
    intent_sweep = band_sweep(
        cache_dir, layers,
        clean_split=args.clean_split,
        wrapped_split=args.wrapped_split,
        norm_stats=norm_stats,
        kill_delta_max=kill_delta_max,
        auroc_floor=auroc_floor,
    )

    print(f"{'layer':>6} {'clean':>8} {'wrapped':>8} {'delta':>8} {'band?':>6}")
    print("-" * 42)
    for entry in intent_sweep["per_layer"]:
        mark = " *" if entry["in_band"] else ""
        print(f"{entry['layer']:>6} {entry['auroc_clean']:>8.3f} {entry['auroc_wrapped']:>8.3f} "
              f"{entry['delta']:>+8.3f} {mark:>6}")
    print(f"\n[kill_test] intent band B = {intent_sweep['band']}")

    # --- Collect axis sweeps ---
    axis_sweeps = {"intent": intent_sweep}

    # Severity axis would run here when severity-labeled data exists.
    # The band_sweep function is axis-agnostic; it just needs binary labels
    # in the cached index.  When severity data arrives, add:
    #   severity_sweep = band_sweep(cache_dir, layers, ..., severity_labels)
    #   axis_sweeps["severity"] = severity_sweep

    # --- Intersection ---
    result = multi_axis_band(axis_sweeps)
    print(f"\n[kill_test] === BAND INTERSECTION ===")
    for axis_name, band in result["per_axis"].items():
        print(f"  {axis_name}: {band}")
    print(f"  intersection: {result['band']}")
    if result["empty_intersection"]:
        print(f"  NOTE: empty intersection — axes may be invariant at different depths")

    # --- Save results ---
    output = {
        "layers_swept": layers,
        "kill_delta_max": kill_delta_max,
        "auroc_floor": auroc_floor,
        "normalized": norm_stats is not None,
        "axes": {},
        "band_intersection": result["band"],
        "empty_intersection": result["empty_intersection"],
    }
    for axis_name, sweep in axis_sweeps.items():
        output["axes"][axis_name] = {
            "per_layer": sweep["per_layer"],
            "band": sweep["band"],
        }

    with open(os.path.join(out, "band_sweep.json"), "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    write_manifest(out, cfg, extra=output)

    print(f"\n[kill_test] results -> {os.path.join(out, 'band_sweep.json')}")

    # --- Verdict ---
    if result["band"]:
        print(f"\n[kill_test] VIABLE — band B = {result['band']} "
              f"({len(result['band'])} layers)")
    elif any(sweep["band"] for sweep in axis_sweeps.values()):
        print(f"\n[kill_test] PARTIAL — individual axis bands exist but intersection is empty")
    else:
        print(f"\n[kill_test] NO BAND — no layer is both discriminative and invariant")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
