#!/usr/bin/env python
"""01_extract - stream residual-stream activations to disk, all layers.

v3: uses output_hidden_states=True for one forward pass per example (not one
per layer).  Extracts the post-block residual stream h_ℓ at all layers in
tap.layer_range (or all layers if layer_range covers the full model).

Runs a smoke check on a handful of prompts first to verify the full
hidden-state stack comes back correctly (shape, device, no silent fallback).

    python scripts/01_extract.py --config configs/v3.yaml
    python scripts/01_extract.py --config configs/v3.yaml --fake-model
    python scripts/01_extract.py --config configs/v3.yaml --smoke-only
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acsl.extract import extract_activations, smoke_check
from acsl.model import assert_frozen
from acsl.norm import compute_norm_stats, save_norm_stats
from acsl.runtime import build_parser, build_examples, layer_range, load_model, resolve, write_manifest


def main() -> int:
    parser = build_parser(__doc__)
    parser.add_argument("--fake-model", action="store_true", help="offline fake model + synthetic conditions")
    parser.add_argument("--limit", type=int, default=None, help="cap examples per condition")
    parser.add_argument("--chat-template", action=argparse.BooleanOptionalAction, default=True,
                        help="wrap prompts in the model's chat template (default: on)")
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False,
                        help="Qwen3 thinking mode in the chat template (default: OFF — documented "
                             "ACSL design + matches serving). MUST match server.py.")
    parser.add_argument("--smoke-only", action="store_true",
                        help="run smoke check only, don't extract")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="skip the pre-extraction smoke check")
    parser.add_argument("--norm-split", default="clean",
                        help="split to compute normalization stats from (default: clean)")
    args = parser.parse_args()
    cfg, out = resolve(args)

    model, tokenizer = load_model(cfg, fake=args.fake_model)
    assert_frozen(model)
    if tokenizer is not None and hasattr(tokenizer, "truncation_side"):
        tokenizer.truncation_side = "left"

    layers = layer_range(cfg)
    n_layers = getattr(getattr(model, "config", object()), "num_hidden_layers", None)
    if n_layers is not None:
        layers = [L for L in layers if L < n_layers] or list(range(n_layers))

    # --- smoke check ---
    if not args.fake_model and not args.skip_smoke:
        print("[extract] running smoke check (5 prompts)...")
        from acsl.model import pick_device
        n_lay, d_mod = smoke_check(model, tokenizer, device=pick_device())
        print(f"[extract] smoke OK: {n_lay} layers, d_model={d_mod}")
        if args.smoke_only:
            return 0

    if args.smoke_only:
        print("[extract] --smoke-only with --fake-model: nothing to smoke-test")
        return 0

    examples = build_examples(cfg, fake=args.fake_model, limit=args.limit)
    cache_dir = cfg.get_path("data.cache_dir", os.path.join("runs", "cache"))

    print(f"[extract] model={'fake' if args.fake_model else cfg.get_path('model.name')} "
          f"layers={layers} -> {cache_dir}")
    print(f"[extract] enable_thinking={args.enable_thinking} (must match server.py serving)")
    manifest = extract_activations(
        model, examples, layers, cache_dir,
        tokenizer=tokenizer,
        chat_template=bool(args.chat_template) and not args.fake_model,
        max_length=1024,
        enable_thinking=bool(args.enable_thinking),
    )
    print(f"[extract] counts per split: {manifest['counts']}")

    # --- compute and save per-layer normalization stats ---
    norm_split = args.norm_split
    if norm_split in manifest["counts"]:
        print(f"[extract] computing per-layer norm stats from split={norm_split!r}...")
        stats = compute_norm_stats(cache_dir, norm_split, layers)
        norm_path = os.path.join(cache_dir, "norm_stats.npz")
        save_norm_stats(stats, norm_path)
        print(f"[extract] norm stats saved to {norm_path}")

        # Sanity: check variance is ~1 after normalization on training split
        from acsl.norm import load_cached_normalized
        import numpy as np
        sample_layer = layers[len(layers) // 2]
        Xn, _, _ = load_cached_normalized(cache_dir, norm_split, sample_layer, stats)
        var = Xn.var(axis=0).mean()
        print(f"[extract] norm sanity (layer {sample_layer}, split={norm_split!r}): "
              f"mean feature variance = {var:.4f} (expect ~1.0)")
    else:
        print(f"[extract] WARNING: norm_split={norm_split!r} not in extracted splits, "
              f"skipping normalization stats")

    data_paths = [
        cfg.get_path("data.harm_path"), cfg.get_path("data.harmless_path"),
        cfg.get_path("data.wrapped_path"), cfg.get_path("data.benign_sensitive_path"),
    ]
    write_manifest(out, cfg, eval_paths=data_paths, extra={
        "cache_dir": cache_dir, "layers": layers,
        "which": "residual_stream", "counts": manifest["counts"],
    })
    print(f"[extract] OK - manifest -> {os.path.join(out, 'manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
