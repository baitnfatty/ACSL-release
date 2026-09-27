#!/usr/bin/env python
"""11_request_actionability — persisted Request-Actionability labeler + independence report.

Fixes the STATUS.md debt "actionability labeler not in repo". Request Actionability is
a REQUEST-SIDE (prompt-position) property: does the prompt solicit operational steps
(1) vs. ask for understanding (0). It reads ONLY the prompt string — never a model
response — so it cannot encode refusal (see docs/axis_definitions.md).

Pipeline:
  1. label_request_actionability(prompt) — the persisted, form-based rule.
  2. Build the RA diff-of-means direction at EVERY cached layer (norm-stats normalized,
     same recipe that produced the saved vectors).
  3. Report, per audited layer: reproduction cos vs the saved actionability vector;
     independence cos vs the Intent direction; and reference baselines
     cos(Intent, Severity) [known-correlated = "too similar" end] and
     cos(saved RA, Intent) [the original axis's independence].
  4. GATE (Matt, 2026-07-04): only if max cos(RA, Intent) over audited layers <= 0.4
     AND --commit is passed, write directions_L*.npz for all layers. Otherwise
     report-only; a high cosine is a finding, not something to paper over.

    python scripts/11_request_actionability.py --config configs/v3.yaml            # report only
    python scripts/11_request_actionability.py --config configs/v3.yaml --commit   # write if gate passes
"""
from __future__ import annotations
import argparse, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from acsl.extract import load_cached
from acsl.norm import load_norm_stats
from acsl.config import load_config

GATE_MAX_COS = 0.40  # Matt's provisional gate: commit only if max cos(RA, Intent) <= this.

# ── The persisted Request-Actionability labeler (prompt-only) ────────────────
_FIRST_2ND = re.compile(r'\bhow\s+(?:do|can|could|would|should|might|shall)\s+(?:i|we|you|one)\b', re.I)
_HOW_TO    = re.compile(r'\bhow\s+to\b', re.I)
_IMPERATIVE = re.compile(r'^\s*(?:help me|give me|show me|walk me through|teach me|write|create|generate|provide|list|make|build|develop|design|draft|compose|outline|explain how|describe how|tell me how)\b', re.I)
_WANT_TO   = re.compile(r"\b(?:i want to|i need to|i'?m trying to)\b", re.I)
_GOODWAYS  = re.compile(r'\bwhat (?:are (?:good|the best)|is the best) ways?\s+(?:to|of|for)\b', re.I)
_STEPS     = re.compile(r'\bsteps?\s+(?:to|for)\b', re.I)

def label_request_actionability(prompt: str) -> int:
    """1 if the REQUEST solicits operational steps (first/second-person action,
    imperative, how-to, "good ways to <verb>", "steps to"); 0 if it asks for
    understanding ("what is", "why", "how did X happen", "how does X work").
    Reads only the prompt; independent of any model response/refusal."""
    p = prompt.strip()
    if (_IMPERATIVE.match(p) or _FIRST_2ND.search(p) or _HOW_TO.search(p)
            or _WANT_TO.search(p) or _GOODWAYS.search(p) or _STEPS.search(p)):
        return 1
    return 0


def _direction(Xn, yy):
    if (yy == 1).sum() == 0 or (yy == 0).sum() == 0:
        return None
    d = Xn[yy == 1].mean(0) - Xn[yy == 0].mean(0)
    n = np.linalg.norm(d)
    return (d / n).astype(np.float32) if n > 0 else None

def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# ── Discriminative test: does RA add harm-detection value beyond Intent? ──────
def _dom(Xn, lab):
    d = Xn[lab == 1].mean(0) - Xn[lab == 0].mean(0)
    n = np.linalg.norm(d)
    return d / n if n > 0 else d

def _loglik(model, Xf, y):
    from numpy import log, clip, sum as nsum
    p = clip(model.predict_proba(Xf)[:, 1], 1e-12, 1 - 1e-12)
    return float(nsum(y * log(p) + (1 - y) * log(1 - p)))

def _fit_lr(Xf, y):
    from sklearn.linear_model import LogisticRegression
    # near-unregularized => ~MLE, so the likelihood-ratio test is valid
    return LogisticRegression(C=1e6, max_iter=5000).fit(Xf, y)

def run_detection(all_layers, cache_dir, ns, rec, cos_by_layer):
    """Per layer: RA-alone / Intent-alone / joint harm-detection AUROC, ΔAUROC,
    and a likelihood-ratio test for RA's incremental value — in two regimes:
    CLEAN (5-fold CV, directions refit per fold) and WRAPPED (kill test:
    directions+probe fit on clean, evaluated on jailbroken prompts).
    Target Y = harm/intent label. Independent of cosine by construction."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from scipy.stats import chi2

    def labels(ids):
        harm = np.array([int(rec[str(i)].get("intent", 0) >= 1) if str(i) in rec else 0 for i in ids])
        ra   = np.array([label_request_actionability(rec[str(i)]["prompt"]) if str(i) in rec else 0 for i in ids])
        return harm, ra

    def norm(cache_dir, split, L):
        X, _, ids = load_cached(cache_dir, split, L)
        mu = ns[L]["mu"]; sig = np.clip(ns[L]["sigma"], 1e-12, None)
        return (X - mu) / sig, ids

    def lrt(pint, pra, y):
        m_r = _fit_lr(pint[:, None], y)
        m_f = _fit_lr(np.c_[pint, pra], y)
        stat = 2 * (_loglik(m_f, np.c_[pint, pra], y) - _loglik(m_r, pint[:, None], y))
        return max(stat, 0.0), float(chi2.sf(max(stat, 0.0), 1))

    print("\n" + "=" * 92)
    print("DISCRIMINATIVE TEST — does Request-Actionability add harm-detection value beyond Intent?")
    print("Target Y = harm/intent label. cos column is GEOMETRY (direction angle), NOT redundancy.")
    print("=" * 92)

    results = {"clean": [], "wrapped": []}
    for regime in ("clean", "wrapped"):
        print(f"\n--- regime: {regime.upper()} "
              + ("(5-fold CV, directions refit per fold)" if regime == "clean"
                 else "(directions+probe fit on CLEAN, evaluated on WRAPPED/jailbroken)") + " ---")
        print(f"{'layer':>5} {'cos(RA,Int)':>11} {'AUC_RA':>7} {'AUC_Int':>8} {'AUC_joint':>10} {'dAUROC':>8} {'LRT_p':>10} {'signif':>7}")
        for L in all_layers:
            Xc, idc = norm(cache_dir, "clean", L)
            hc, rc = labels(idc)
            if regime == "clean":
                skf = StratifiedKFold(5, shuffle=True, random_state=0)
                oi, ora, oj = np.zeros(len(hc)), np.zeros(len(hc)), np.zeros(len(hc))
                for tr, te in skf.split(Xc, hc):
                    ui = _dom(Xc[tr], hc[tr])   # intent direction, fit on train fold
                    ur = _dom(Xc[tr], rc[tr])   # RA direction, fit on train fold
                    pit, prt = Xc[tr] @ ui, Xc[tr] @ ur
                    pie, pre = Xc[te] @ ui, Xc[te] @ ur
                    oi[te]  = _fit_lr(pit[:, None], hc[tr]).predict_proba(pie[:, None])[:, 1]
                    ora[te] = _fit_lr(prt[:, None], hc[tr]).predict_proba(pre[:, None])[:, 1]
                    oj[te]  = _fit_lr(np.c_[pit, prt], hc[tr]).predict_proba(np.c_[pie, pre])[:, 1]
                a_ra, a_int, a_j = roc_auc_score(hc, ora), roc_auc_score(hc, oi), roc_auc_score(hc, oj)
                # LRT on full clean
                ui, ur = _dom(Xc, hc), _dom(Xc, rc)
                stat, p = lrt(Xc @ ui, Xc @ ur, hc)
            else:
                ui, ur = _dom(Xc, hc), _dom(Xc, rc)
                pit, prt = Xc @ ui, Xc @ ur
                Xw, idw = norm(cache_dir, "wrapped", L)
                hw, _ = labels(idw)
                piw, prw = Xw @ ui, Xw @ ur
                a_int = roc_auc_score(hw, _fit_lr(pit[:, None], hc).predict_proba(piw[:, None])[:, 1])
                a_ra  = roc_auc_score(hw, _fit_lr(prt[:, None], hc).predict_proba(prw[:, None])[:, 1])
                a_j   = roc_auc_score(hw, _fit_lr(np.c_[pit, prt], hc).predict_proba(np.c_[piw, prw])[:, 1])
                stat, p = lrt(piw, prw, hw)   # LRT on wrapped
            dauc = a_j - a_int
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            results[regime].append((L, dauc, p))
            print(f"L{L:<4} {cos_by_layer[L]:>+11.3f} {a_ra:>7.3f} {a_int:>8.3f} {a_j:>10.3f} {dauc:>+8.3f} {p:>10.2e} {sig:>7}")

    print("\n" + "=" * 92)
    for regime in ("clean", "wrapped"):
        sig_layers = [(L, d, p) for (L, d, p) in results[regime] if p < 0.05 and d > 0]
        best = max(results[regime], key=lambda t: t[1])
        print(f"[{regime.upper()}] layers where RA adds signif. positive incremental detection (p<0.05, dAUROC>0): "
              f"{[L for L,_,_ in sig_layers] or 'NONE'}")
        print(f"           best dAUROC = {best[1]:+.3f} @ L{best[0]} (p={best[2]:.1e})")
    print("\nNOTE: incremental ΔAUROC alone is NOT evidence RA is special — a collinear 2nd axis can "
          "cancel a common-mode wrapper offset. See CONTROL table (--controls) for RA vs random vs severity.")


def run_controls(all_layers, cache_dir, ns, rec, cos_by_layer, n_rand=15, seed=0):
    """Rule-9 control (2026-07-05): the WRAPPED ΔAUROC(Intent+RA) is uninterpretable
    without a null. Compare RA's incremental AUROC against (i) Intent+Severity and
    (ii) Intent+RANDOM diff-of-means features (n_rand trials, matched positive rate).
    RA is 'special' at a layer ONLY if its ΔAUROC exceeds the random p95 there."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.RandomState(seed)

    def norm(split, L):
        X, _, ids = load_cached(cache_dir, split, L)
        mu = ns[L]["mu"]; sig = np.clip(ns[L]["sigma"], 1e-12, None)
        return (X - mu) / sig, ids

    def harm_of(ids):
        return np.array([int(rec[str(i)].get("intent", 0) >= 1) if str(i) in rec else 0 for i in ids])
    def ra_of(ids):
        return np.array([label_request_actionability(rec[str(i)]["prompt"]) if str(i) in rec else 0 for i in ids])
    def sev_of(ids):
        return np.array([int(rec[str(i)].get("severity", 0) >= 1) if str(i) in rec else 0 for i in ids])

    def dauroc(Xc, hc, Xw, hw, second_lab):
        ui = _dom(Xc, hc); us = _dom(Xc, second_lab)
        pic, psc = Xc @ ui, Xc @ us
        piw, psw = Xw @ ui, Xw @ us
        a_int = roc_auc_score(hw, _fit_lr(pic[:, None], hc).predict_proba(piw[:, None])[:, 1])
        a_j = roc_auc_score(hw, _fit_lr(np.c_[pic, psc], hc).predict_proba(np.c_[piw, psw])[:, 1])
        return a_j - a_int

    print("\n" + "=" * 92)
    print("CONTROL (WRAPPED regime): is RA's incremental ΔAUROC special, or generic collinear-suppressor?")
    print("RA is 'special' only where ΔAUROC(RA) > ΔAUROC(random p95). Random = matched-positive-rate diff-of-means.")
    print("=" * 92)
    print(f"{'layer':>5} {'cos(RA,Int)':>11} {'dAUC_RA':>8} {'dAUC_Sev':>9} {'rand_mean':>9} {'rand_p95':>9} {'RA>rand?':>9}")
    special = []
    for L in all_layers:
        Xc, idc = norm("clean", L); hc = harm_of(idc)
        Xw, idw = norm("wrapped", L); hw = harm_of(idw)
        npos = int(ra_of(idc).sum())
        d_ra = dauroc(Xc, hc, Xw, hw, ra_of(idc))
        d_sev = dauroc(Xc, hc, Xw, hw, sev_of(idc))
        rnd = []
        for _ in range(n_rand):
            lab = np.zeros(len(hc), dtype=int); lab[rng.choice(len(hc), npos, replace=False)] = 1
            rnd.append(dauroc(Xc, hc, Xw, hw, lab))
        rnd = np.array(rnd)
        p95 = float(np.percentile(rnd, 95))
        is_special = d_ra > p95
        if is_special:
            special.append(L)
        print(f"L{L:<4} {cos_by_layer[L]:>+11.3f} {d_ra:>+8.3f} {d_sev:>+9.3f} {rnd.mean():>+9.3f} {p95:>+9.3f} {'YES' if is_special else 'no':>9}")
    print("=" * 92)
    print(f"Layers where RA beats the random-feature null (p95): {special or 'NONE'}")
    print("If NONE: RA's ΔAUROC is generic collinear-suppressor / wrapper-cancellation, NOT RA-specific harm signal.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--commit", action="store_true", help="write npz for all layers if gate passes")
    ap.add_argument("--detect", action="store_true", help="run the discriminative ΔAUROC + LRT test")
    ap.add_argument("--controls", action="store_true", help="RA vs random/severity 2nd-feature null (Rule-9 control)")
    ap.add_argument("--audit-dir", default=None, help="dir with saved directions_L*.npz")
    args = ap.parse_args()
    cfg = load_config(args.config)

    cache_dir = cfg.get_path("data.cache_dir")
    ns = load_norm_stats(os.path.join(cache_dir, "norm_stats.npz"))
    audit_dir = args.audit_dir or ("runs/10_direction_audit_1.7b" if "cache_v3_4b" not in cache_dir
                                   else "runs/10_direction_audit_4b")

    # id -> prompt, intent label, severity label
    rec = {}
    for p in ["data/v3_harmful.jsonl", "data/v3_harmless.jsonl", "data/v3_wrapped.jsonl"]:
        if os.path.exists(p):
            for line in open(p):
                r = json.loads(line); rec[r["id"]] = r

    # audited layers = those with a saved actionability vector
    audited = []
    saved_ra = {}
    for fn in sorted(os.listdir(audit_dir)):
        m = re.match(r"directions_L(\d+)\.npz$", fn)
        if m:
            L = int(m.group(1))
            z = np.load(os.path.join(audit_dir, fn))
            if "actionability" in z:
                audited.append(L); saved_ra[L] = z["actionability"]
    audited.sort()

    # cached layers: use config tap range, clamp to what actually has data
    lo, hi = cfg.get_path("tap.layer_range", [0, 27])
    all_layers = []
    for L in range(int(lo), int(hi) + 1):
        X, _, _ = load_cached(cache_dir, "clean", L)
        if X.shape[0] > 0:
            all_layers.append(L)

    print(f"cache={cache_dir}  layers cached={len(all_layers)}  audited={audited}")
    ra_dirs, intent_dirs, sev_dirs = {}, {}, {}
    npos = nneg = 0
    for L in all_layers:
        X, y, ids = load_cached(cache_dir, "clean", L)          # one load per layer
        mu = ns[L]["mu"]; sig = np.clip(ns[L]["sigma"], 1e-12, None); Xn = (X - mu) / sig
        ra_lab  = np.array([label_request_actionability(rec[str(i)]["prompt"]) if str(i) in rec else 0 for i in ids])
        int_lab = np.array([int(rec[str(i)].get("intent", 0) >= 1) if str(i) in rec else 0 for i in ids])
        sev_lab = np.array([int(rec[str(i)].get("severity", 0) >= 1) if str(i) in rec else 0 for i in ids])
        ra_dirs[L]     = _direction(Xn, ra_lab)
        intent_dirs[L] = _direction(Xn, int_lab)
        sev_dirs[L]    = _direction(Xn, sev_lab)
        npos = int((ra_lab == 1).sum()); nneg = int((ra_lab == 0).sum())

    print(f"\nRequest-Actionability labeler: pos={npos} neg={nneg} of {npos+nneg}\n")
    print(f"{'layer':>5} {'reproduce':>10} {'cos(RA,Int)':>12} {'ref cos(Int,Sev)':>16} {'ref cos(savedRA,Int)':>20}")
    print("-" * 70)
    gate_vals = []
    for L in audited:
        repro = _cos(ra_dirs[L], saved_ra[L])
        ra_int = _cos(ra_dirs[L], intent_dirs[L])
        int_sev = _cos(intent_dirs[L], sev_dirs[L])
        saved_int = _cos(saved_ra[L], intent_dirs[L])
        gate_vals.append(ra_int)
        print(f"L{L:<4} {repro:>+10.4f} {ra_int:>+12.4f} {int_sev:>+16.4f} {saved_int:>+20.4f}")

    # full-depth independence curve (report-only; the finding is where axes separate)
    print(f"\n=== independence-vs-depth (all {len(all_layers)} layers) ===")
    print(f"{'layer':>5} {'cos(RA,Int)':>12} {'cos(Int,Sev)':>13}")
    for L in all_layers:
        print(f"L{L:<4} {_cos(ra_dirs[L], intent_dirs[L]):>+12.4f} {_cos(intent_dirs[L], sev_dirs[L]):>+13.4f}")

    if args.detect or args.controls:
        cos_by_layer = {L: _cos(ra_dirs[L], intent_dirs[L]) for L in all_layers}
        if args.detect:
            run_detection(all_layers, cache_dir, ns, rec, cos_by_layer)
        if args.controls:
            run_controls(all_layers, cache_dir, ns, rec, cos_by_layer)

    max_cos = max(gate_vals)
    print("\n" + "=" * 70)
    print(f"GATE: max cos(RA, Intent) over audited layers = {max_cos:+.4f}  (threshold {GATE_MAX_COS})")
    passed = max_cos <= GATE_MAX_COS
    print(f"GATE {'PASS' if passed else 'FAIL'} — " +
          ("independent enough; safe to commit + sweep." if passed else
           "TOO SIMILAR: request can't distinguish intent from actionability. This is a FINDING. Not committing."))

    if args.commit and passed:
        written = 0
        for L in all_layers:
            path = os.path.join(audit_dir, f"directions_L{L}.npz")
            existing = dict(np.load(path)) if os.path.exists(path) else {}
            existing["actionability"] = ra_dirs[L]
            existing.setdefault("intent", intent_dirs[L])
            np.savez(path, **existing)
            written += 1
        print(f"\n[commit] wrote/updated actionability in {written} directions_L*.npz under {audit_dir}")
    elif args.commit and not passed:
        print("\n[commit] REFUSED — gate failed. No files written.")
    else:
        print("\n[report-only] no files written. Re-run with --commit to write if the gate passes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
