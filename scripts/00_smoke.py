#!/usr/bin/env python
"""00_smoke - end-to-end wiring check on the harmless synthetic set.

Loads a small frozen model, taps one layer to extract o_L, trains a dummy head
for 1 epoch on the cached activations, computes the diff-of-means caution
direction, runs the security loop once, and prints the chosen action.

No network calls to harmful data. No .backward() through the base model - only
the SecurityHead is trained. Use --fake-model to run fully offline (no download).

    python scripts/00_smoke.py --config configs/default.yaml --fake-model
"""

from __future__ import annotations

import os
import sys

import torch

# Allow running as a plain script (python scripts/00_smoke.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acsl.directions import diff_of_means
from acsl.extract import extract_activations, load_cached
from acsl.head import train_head, risk_logit
from acsl.hooks import AttnTap
from acsl.loop import security_loop
from acsl.model import assert_frozen
from acsl.policy import action_policy
from acsl.runtime import build_parser, resolve, write_manifest
from acsl.data.synthetic_smoke import synthetic_smoke_examples


def main() -> int:
    parser = build_parser(__doc__)
    parser.add_argument("--fake-model", action="store_true", help="use the offline fake model (no download)")
    parser.add_argument("--layer", type=int, default=None, help="layer to tap (default: middle of tap.layer_range)")
    args = parser.parse_args()
    cfg, out = resolve(args)

    # --- load a frozen base ------------------------------------------------
    if args.fake_model:
        from acsl._fake import load_fake_model

        model, tokenizer = load_fake_model(d_model=64, n_layers=6)
        print("[smoke] loaded FAKE frozen model (offline)")
    else:
        from acsl.model import load_frozen_model

        model, tokenizer = load_frozen_model(cfg.get_path("model.name"), cfg.get_path("model.dtype", "float16"))
        print(f"[smoke] loaded frozen model: {cfg.get_path('model.name')}")
    assert_frozen(model)

    lo, hi = cfg.get_path("tap.layer_range", [0, 0])
    layer = args.layer if args.layer is not None else (lo + hi) // 2
    n_layers = getattr(getattr(model, "config", object()), "num_hidden_layers", None)
    if n_layers is not None and layer >= n_layers:
        layer = n_layers // 2
    print(f"[smoke] tapping layer {layer}")

    examples = synthetic_smoke_examples(n_per_class=8, split="smoke")

    # --- show a raw o_L capture (batch, seq, d_model) ----------------------
    device = next(model.parameters()).device
    print(f"[smoke] device = {device}")
    enc = tokenizer(examples[0].prompt, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad(), AttnTap(model, layer) as tap:
        model(**enc)
        print(f"[smoke] o_L shape = {tuple(tap.o_L.shape)}  (batch, seq, d_model)")
        assert tap.o_L.dim() == 3 and tap.n_calls == 1

    # --- stream activations to disk ----------------------------------------
    cache_dir = os.path.join(out, "cache")
    manifest = extract_activations(model, examples, [layer], cache_dir, tokenizer=tokenizer, which="o_L")
    print(f"[smoke] extracted {manifest['counts']} -> {cache_dir}")

    # --- train a dummy head for 1 epoch ------------------------------------
    smoke_cfg = dict(cfg)
    smoke_cfg["head"] = {**cfg.get_path("head", {}), "n_categories": 2, "epochs": 1}
    head, val_auroc = train_head(cache_dir, None, layer, smoke_cfg)
    print(f"[smoke] trained head (1 epoch); val AUROC = {val_auroc:.3f}")
    assert_frozen(model)  # base still frozen after training the head

    # --- diff-of-means caution direction -----------------------------------
    X, y, _ = load_cached(cache_dir, "smoke", layer)
    Xt = torch.as_tensor(X, dtype=torch.float32)
    u = diff_of_means(Xt[y == 1], Xt[y == 0])
    print(f"[smoke] caution direction u: shape {tuple(u.shape)}, ||u|| = {float(u.norm()):.3f}")

    # --- run the loop once + pick an action --------------------------------
    h0 = Xt[0]
    h_adj, r_star, safe = security_loop(
        h0, head, u,
        g_max=float(cfg.get_path("loop.g_max", 8.0)),
        K=int(cfg.get_path("loop.K", 4)),
        eps=float(cfg.get_path("loop.eps", 1e-2)),
        thr=float(cfg.get_path("loop.thr", 0.0)),
    )
    action = action_policy(r_star, safe)
    print(f"[smoke] loop -> r_star={r_star:.3f} safe={safe} | action={action!r}")

    write_manifest(out, cfg, eval_paths=[], extra={"smoke": {"layer": layer, "val_auroc": val_auroc, "action": action}})
    print(f"[smoke] OK - manifest written to {os.path.join(out, 'manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
