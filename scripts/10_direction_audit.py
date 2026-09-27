#!/usr/bin/env python
"""10_direction_audit — Items 1, 3, 5, 6 from the interpretability checklist.

Extracts activations for new prompts (diverse corpus + minimal pairs),
projects onto intent/severity directions, and reports:
  - Spectrum analysis with domain breakdown (item 1)
  - Minimal pair divergence (item 3)
  - Steering asymmetry: which benign prompts flip under +alpha*u (item 6)
  - Sufficiency/completeness summary (item 5)

    python scripts/10_direction_audit.py --config configs/v3_4b.yaml
"""
from __future__ import annotations

import json
import os
import sys
import gc

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acsl.model import load_frozen_model, pick_device
from acsl.norm import load_norm_stats
from acsl.extract import load_cached


def _extract_single(model, tokenizer, prompt, device, max_length=1024):
    """Extract per-layer last-token activations for one prompt."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=False,
    )
    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=max_length, add_special_tokens=False)
    inputs = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    hidden = out.hidden_states  # tuple of (1, seq, d_model), len = n_layers+1
    attn_mask = inputs.get("attention_mask")
    if attn_mask is not None:
        last_idx = int(attn_mask.sum() - 1)
    else:
        last_idx = enc["input_ids"].shape[1] - 1

    acts = {}
    for layer_idx in range(1, len(hidden)):
        L = layer_idx - 1
        h = hidden[layer_idx][0, last_idx, :].cpu().float().numpy()
        acts[L] = h

    del out, hidden, inputs
    gc.collect()
    return acts


def _compute_direction(cache_dir, layer, norm_stats, label_fn=None):
    """Compute diff-of-means direction from cached clean data."""
    X, y, ids = load_cached(cache_dir, "clean", layer)
    if norm_stats and layer in norm_stats:
        mu = norm_stats[layer]["mu"]
        sigma = np.clip(norm_stats[layer]["sigma"], 1e-12, None)
        X = (X - mu) / sigma

    if label_fn is not None:
        y = np.array([label_fn(str(i)) for i in ids])

    pos = X[y == 1]
    neg = X[y == 0]
    diff = pos.mean(0) - neg.mean(0)
    u = diff / np.linalg.norm(diff)
    return u.astype(np.float32)


def _normalize(h, layer, norm_stats):
    """Apply train-fit normalization to a single activation vector."""
    if norm_stats and layer in norm_stats:
        mu = norm_stats[layer]["mu"]
        sigma = np.clip(norm_stats[layer]["sigma"], 1e-12, None)
        return (h - mu) / sigma
    return h


def main() -> int:
    from acsl.runtime import build_parser, resolve, write_manifest
    parser = build_parser(__doc__)
    parser.add_argument("--layers", nargs="+", type=int, default=[33, 17])
    parser.add_argument("--alpha", nargs="+", type=float, default=[5.0, 10.0, 20.0])
    args = parser.parse_args()
    cfg, out = resolve(args)

    cache_dir = cfg.get_path("data.cache_dir", os.path.join("runs", "cache"))
    norm_path = os.path.join(cache_dir, "norm_stats.npz")
    norm_stats = load_norm_stats(norm_path) if os.path.exists(norm_path) else None

    # Load severity map for severity direction
    sev_map = {}
    for path in ["data/v3_harmful.jsonl", "data/v3_harmless.jsonl", "data/v3_wrapped.jsonl"]:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    sev_map[r["id"]] = r.get("severity", 0)

    sev_label_fn = lambda i: 1 if sev_map.get(i, 0) >= 1 else 0

    # Compute directions at each layer
    directions = {}
    for L in args.layers:
        u_intent = _compute_direction(cache_dir, L, norm_stats)
        u_sev = _compute_direction(cache_dir, L, norm_stats, label_fn=sev_label_fn)
        directions[L] = {"intent": u_intent, "severity": u_sev}
        cos = float(u_intent @ u_sev)
        print(f"[L{L}] cos(intent, severity) = {cos:.4f}")

    # Load model for new prompt extraction
    print("\nLoading model...")
    model, tokenizer = load_frozen_model(
        cfg.get_path("model.name"), device=pick_device(),
        dtype=cfg.get_path("model.dtype", "float32"),
    )
    device = next(model.parameters()).device

    # Load diverse corpus
    diverse = []
    if os.path.exists("data/diverse_corpus.jsonl"):
        with open("data/diverse_corpus.jsonl") as f:
            diverse = [json.loads(l) for l in f]
    print(f"Loaded {len(diverse)} diverse corpus prompts")

    # Load minimal pairs
    pairs = []
    if os.path.exists("data/minimal_pairs.jsonl"):
        with open("data/minimal_pairs.jsonl") as f:
            pairs = [json.loads(l) for l in f]
    print(f"Loaded {len(pairs)} minimal pairs")

    # ===== EXTRACT & PROJECT DIVERSE CORPUS =====
    print("\n=== DIVERSE CORPUS EXTRACTION & PROJECTION ===")
    diverse_results = []
    for i, ex in enumerate(diverse):
        acts = _extract_single(model, tokenizer, ex["prompt"], device)
        scores = {}
        for L in args.layers:
            h_norm = _normalize(acts[L], L, norm_stats)
            scores[L] = {
                "intent": float(h_norm @ directions[L]["intent"]),
                "severity": float(h_norm @ directions[L]["severity"]),
            }
        diverse_results.append({**ex, "scores": scores})
        if (i + 1) % 10 == 0:
            print(f"  extracted {i+1}/{len(diverse)}")
        del acts; gc.collect()

    # Report by domain
    for L in args.layers:
        print(f"\n--- Layer {L}: Diverse corpus by domain ---")
        domains = {}
        for r in diverse_results:
            d = r.get("domain", "?")
            domains.setdefault(d, []).append(r["scores"][L])

        print(f"{'domain':>15} {'n':>4} {'intent_mean':>12} {'intent_std':>11} {'sev_mean':>10} {'sev_std':>9} {'concern?':>9}")
        print("-" * 80)
        for d in sorted(domains):
            si = [s["intent"] for s in domains[d]]
            ss = [s["severity"] for s in domains[d]]
            mi, msi = np.mean(si), np.std(si)
            ms, mss = np.mean(ss), np.std(ss)
            # Flag domains where mean score is above 0 (positive = harm direction)
            flag = ""
            if mi > 0: flag += "I!"
            if ms > 0: flag += "S!"
            print(f"{d:>15} {len(si):>4} {mi:>+12.2f} {msi:>11.2f} {ms:>+10.2f} {mss:>9.2f} {flag:>9}")

        # Individual high-scorers
        print(f"\n  Top 10 by intent score:")
        by_intent = sorted(diverse_results, key=lambda r: -r["scores"][L]["intent"])
        for r in by_intent[:10]:
            s = r["scores"][L]
            print(f"    I={s['intent']:>+7.2f} S={s['severity']:>+7.2f}  [{r['domain']:>12}] {r['prompt'][:70]}")

        print(f"\n  Top 10 by severity score:")
        by_sev = sorted(diverse_results, key=lambda r: -r["scores"][L]["severity"])
        for r in by_sev[:10]:
            s = r["scores"][L]
            print(f"    I={s['intent']:>+7.2f} S={s['severity']:>+7.2f}  [{r['domain']:>12}] {r['prompt'][:70]}")

    # ===== MINIMAL PAIRS =====
    print("\n\n=== MINIMAL PAIR ANALYSIS ===")
    pair_results = []
    for p in pairs:
        acts_a = _extract_single(model, tokenizer, p["a"]["prompt"], device)
        acts_b = _extract_single(model, tokenizer, p["b"]["prompt"], device)
        scores = {}
        for L in args.layers:
            ha = _normalize(acts_a[L], L, norm_stats)
            hb = _normalize(acts_b[L], L, norm_stats)
            scores[L] = {
                "a_intent": float(ha @ directions[L]["intent"]),
                "a_severity": float(ha @ directions[L]["severity"]),
                "b_intent": float(hb @ directions[L]["intent"]),
                "b_severity": float(hb @ directions[L]["severity"]),
            }
        pair_results.append({**p, "scores": scores})
        del acts_a, acts_b; gc.collect()

    for L in args.layers:
        print(f"\n--- Layer {L}: Minimal pairs ---")
        print(f"{'pair_id':>16} {'type':>25} | {'a_I':>7} {'a_S':>7} | {'b_I':>7} {'b_S':>7} | {'dI':>7} {'dS':>7} | separates?")
        print("-" * 120)
        n_sep_intent = 0
        n_sep_sev = 0
        n_total = 0
        for pr in pair_results:
            s = pr["scores"][L]
            di = s["b_intent"] - s["a_intent"]
            ds = s["b_severity"] - s["a_severity"]
            # Does intent separate? (b should be higher for harmful)
            sep_i = "Y" if di > 3.0 else ("~" if di > 0 else "N")
            sep_s = "Y" if ds > 3.0 else ("~" if ds > 0 else "N")
            if di > 3.0: n_sep_intent += 1
            if ds > 3.0: n_sep_sev += 1
            n_total += 1
            print(f"{pr['pair_id']:>16} {pr['type']:>25} | {s['a_intent']:>+7.1f} {s['a_severity']:>+7.1f} | {s['b_intent']:>+7.1f} {s['b_severity']:>+7.1f} | {di:>+7.1f} {ds:>+7.1f} | I:{sep_i} S:{sep_s}")

        print(f"\n  Intent separates {n_sep_intent}/{n_total} pairs (>{3.0} gap)")
        print(f"  Severity separates {n_sep_sev}/{n_total} pairs (>{3.0} gap)")

        # Same-surface pairs: does the probe separate despite shared surface?
        same_surface = [p for p in pair_results if p["type"] == "same_surface_diff_concept"]
        topic_act = [p for p in pair_results if p["type"] == "topic_vs_act"]
        if same_surface:
            gaps_i = [p["scores"][L]["b_intent"] - p["scores"][L]["a_intent"] for p in same_surface]
            gaps_s = [p["scores"][L]["b_severity"] - p["scores"][L]["a_severity"] for p in same_surface]
            print(f"\n  Same-surface pairs (n={len(same_surface)}): intent gap mean={np.mean(gaps_i):+.1f} std={np.std(gaps_i):.1f}, sev gap mean={np.mean(gaps_s):+.1f} std={np.std(gaps_s):.1f}")
        if topic_act:
            gaps_i = [p["scores"][L]["b_intent"] - p["scores"][L]["a_intent"] for p in topic_act]
            gaps_s = [p["scores"][L]["b_severity"] - p["scores"][L]["a_severity"] for p in topic_act]
            print(f"  Topic-vs-act pairs (n={len(topic_act)}): intent gap mean={np.mean(gaps_i):+.1f} std={np.std(gaps_i):.1f}, sev gap mean={np.mean(gaps_s):+.1f} std={np.std(gaps_s):.1f}")

    # ===== STEERING ASYMMETRY (Item 6) =====
    print("\n\n=== STEERING ASYMMETRY ===")
    print("Adding +alpha*u to benign prompt activations, checking which cross threshold")

    # Get threshold from cached data (top of benign distribution)
    for L in args.layers:
        X_clean, y_clean, _ = load_cached(cache_dir, "clean", L)
        if norm_stats and L in norm_stats:
            mu = norm_stats[L]["mu"]
            sigma = np.clip(norm_stats[L]["sigma"], 1e-12, None)
            X_clean = (X_clean - mu) / sigma

        benign_scores_i = (X_clean[y_clean == 0] @ directions[L]["intent"])
        benign_scores_s = (X_clean[y_clean == 0] @ directions[L]["severity"])
        harm_scores_i = (X_clean[y_clean == 1] @ directions[L]["intent"])
        harm_scores_s = (X_clean[y_clean == 1] @ directions[L]["severity"])

        # Threshold at ~5% over-refusal (95th percentile of benign)
        thresh_i = float(np.percentile(benign_scores_i, 95))
        thresh_s = float(np.percentile(benign_scores_s, 95))

        print(f"\n--- Layer {L} ---")
        print(f"  Intent threshold (p95 benign): {thresh_i:+.2f}")
        print(f"  Severity threshold (p95 benign): {thresh_s:+.2f}")
        print(f"  Harmful intent mean: {harm_scores_i.mean():+.2f}, severity mean: {harm_scores_s.mean():+.2f}")

        for alpha in args.alpha:
            print(f"\n  alpha={alpha}:")
            flipped_i = []
            flipped_s = []
            for r in diverse_results:
                base_i = r["scores"][L]["intent"]
                base_s = r["scores"][L]["severity"]
                steered_i = base_i + alpha * float(np.linalg.norm(directions[L]["intent"]))
                steered_s = base_s + alpha * float(np.linalg.norm(directions[L]["severity"]))

                if base_i < thresh_i and steered_i >= thresh_i:
                    flipped_i.append(r)
                if base_s < thresh_s and steered_s >= thresh_s:
                    flipped_s.append(r)

            print(f"    Intent flipped: {len(flipped_i)}/{len(diverse_results)}")
            if flipped_i:
                domain_counts = {}
                for r in flipped_i:
                    domain_counts[r.get("domain", "?")] = domain_counts.get(r.get("domain", "?"), 0) + 1
                print(f"    By domain: {dict(sorted(domain_counts.items(), key=lambda x: -x[1]))}")
                for r in flipped_i[:5]:
                    print(f"      base_I={r['scores'][L]['intent']:+.1f} [{r['domain']}] {r['prompt'][:60]}")

            print(f"    Severity flipped: {len(flipped_s)}/{len(diverse_results)}")
            if flipped_s:
                domain_counts = {}
                for r in flipped_s:
                    domain_counts[r.get("domain", "?")] = domain_counts.get(r.get("domain", "?"), 0) + 1
                print(f"    By domain: {dict(sorted(domain_counts.items(), key=lambda x: -x[1]))}")
                for r in flipped_s[:5]:
                    print(f"      base_S={r['scores'][L]['severity']:+.1f} [{r['domain']}] {r['prompt'][:60]}")

    # Save all results
    output = {
        "layers": args.layers,
        "diverse_results": diverse_results,
        "pair_results": [{k: v for k, v in p.items() if k != "scores"} | {"scores": p["scores"]}
                         for p in pair_results],
    }
    out_path = os.path.join(out, "direction_audit.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
