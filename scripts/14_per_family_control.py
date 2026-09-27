#!/usr/bin/env python
"""14_per_family_control — does RA rescue Intent PER jailbreak family, beyond nulls?

Runs ONLY on corrected (thinking-OFF) caches. Per wrapper family:
  positives = that family's WRAPPED-HARMFUL prompts (n=35);
  negatives = the 470 harm==0 examples in the "wrapped" cache split.
Target Y = harm/intent label. Directions (Intent, RA, length, random) fit on CLEAN;
logistic probe fit on CLEAN; AUROC evaluated on this positives+negatives set.

⚠ DATA LIMITATION (found via Rule 9, 2026-07-05): the negatives are NOT wrapped-benign —
`data/v3_wrapped.jsonl` contains 420 wrapped-HARMFUL and ZERO benign, so all 470 negatives
are UNWRAPPED clean benigns (wrapper_family=None, e.g. "How can I kill a Python process?").
The contrast therefore confounds WRAPPING (jailbreak text) with HARM, and no family-labeled
benigns exist, so this test cannot cleanly isolate PER-FAMILY harm signal. Read the results
as "wrapped-harmful-of-family-F vs unwrapped-clean-benign", not a pure per-family harm test.

Per family we report ΔAUROC over Intent-alone for:
  - Intent+RA              (the claim)
  - Intent+random (p95)    (generic collinear-suppressor null, N trials)
  - Intent+length-dir      (length-matched null: a pure length axis at the RA layer)

Decision: RA earns its place on a family iff ΔAUROC(RA) exceeds BOTH nulls there.
Models reported SEPARATELY (never aggregated); 4B@L33 is the more trustworthy read
given the 1.7B@L9 length confound (STATUS 2026-07-05).

    python scripts/14_per_family_control.py --config configs/v3_thinkoff.yaml
    python scripts/14_per_family_control.py --config configs/v3_4b_thinkoff.yaml
"""
from __future__ import annotations
import argparse, importlib.util, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from acsl.extract import load_cached
from acsl.norm import load_norm_stats
from acsl.config import load_config

_spec = importlib.util.spec_from_file_location(
    "ra11", os.path.join(os.path.dirname(os.path.abspath(__file__)), "11_request_actionability.py"))
_ra11 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_ra11)
label_request_actionability = _ra11.label_request_actionability

OP = {"cache_v3": {"intent": 9, "ra": 9}, "cache_v3_4b": {"intent": 33, "ra": 20}}
N_RAND = 15


def _dom(Xn, lab):
    d = Xn[lab == 1].mean(0) - Xn[lab == 0].mean(0)
    n = np.linalg.norm(d)
    return d / n if n > 0 else d

def _fit_lr(Xf, y):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=1e6, max_iter=5000).fit(Xf, y)

def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-rand", type=int, default=N_RAND,
                    help="random-null trials (use 1000 for a stable p95)")
    args = ap.parse_args()
    n_rand = args.n_rand
    cfg = load_config(args.config)
    cache = cfg.get_path("data.cache_dir")
    key = "cache_v3_4b" if "cache_v3_4b" in cache else "cache_v3"
    Li, Lr = OP[key]["intent"], OP[key]["ra"]
    ns = load_norm_stats(os.path.join(cache, "norm_stats.npz"))

    rec = {}
    for p in ["data/v3_harmful.jsonl", "data/v3_harmless.jsonl", "data/v3_wrapped.jsonl"]:
        for line in open(p):
            r = json.loads(line); rec[r["id"]] = r

    def norm(split, L):
        X, _, ids = load_cached(cache, split, L)
        mu = ns[L]["mu"]; sig = np.clip(ns[L]["sigma"], 1e-12, None)
        return (X - mu) / sig, [str(i) for i in ids]

    # ---- CLEAN: directions + probe fit ----
    Xci, idc = norm("clean", Li)
    Xcr, idcr = norm("clean", Lr)
    assert idc == idcr, "clean id order differs across layers"
    harm_c = np.array([int(rec[i].get("intent", 0) >= 1) for i in idc])
    ra_c   = np.array([label_request_actionability(rec[i]["prompt"]) for i in idc])
    wc     = np.array([len(rec[i]["prompt"].split()) for i in idc])          # word count
    len_c  = (wc > np.median(wc)).astype(int)                                 # length label

    u_int = _dom(Xci, harm_c)
    u_ra  = _dom(Xcr, ra_c)
    u_len = _dom(Xcr, len_c)
    rng = np.random.RandomState(0)
    npos = int(ra_c.sum())
    u_rand = []
    for _ in range(n_rand):
        lab = np.zeros(len(harm_c), int); lab[rng.choice(len(harm_c), npos, replace=False)] = 1
        u_rand.append(_dom(Xcr, lab))

    pint_c = Xci @ u_int
    M_int  = _fit_lr(pint_c[:, None], harm_c)
    M_ra   = _fit_lr(np.c_[pint_c, Xcr @ u_ra], harm_c)
    M_len  = _fit_lr(np.c_[pint_c, Xcr @ u_len], harm_c)
    M_rand = [_fit_lr(np.c_[pint_c, Xcr @ u], harm_c) for u in u_rand]

    # ---- WRAPPED: per-family eval ----
    Xwi, idw = norm("wrapped", Li)
    Xwr, idwr = norm("wrapped", Lr)
    assert idw == idwr
    harm_w = np.array([int(rec[i].get("intent", 0) >= 1) for i in idw])
    fam_w  = np.array([rec[i].get("wrapper_family", None) for i in idw])
    neg = harm_w == 0   # NOTE: these are UNWRAPPED clean benigns (all wrapper_family=None),
                        # NOT wrapped-benign — no benign wrapped prompts exist. See docstring.
    families = sorted({f for f in fam_w if f})

    pint_w = Xwi @ u_int
    pra_w  = Xwr @ u_ra
    plen_w = Xwr @ u_len
    prand_w = [Xwr @ u for u in u_rand]

    print(f"\n{'='*100}")
    print(f"PER-FAMILY RESCUE CONTROL — {args.config}  (Intent@L{Li}, RA@L{Lr})  "
          f"neg=UNWRAPPED-clean-benign(n={int(neg.sum())}; confounds wrapping w/ harm — see docstring)")
    print("Target Y=harm. dAUROC = joint − Intent-alone, on wrapped. RA 'wins' iff dRA > max(rand_p95, len).")
    print(f"{'='*100}")
    print("Reporting RA vs null CENTER (percentile of dRA within the random-null dist) AND the p95 bar.")
    print(f"{'family':>20} {'n+':>4} {'AUC_int':>8} {'dRA':>7} {'RA_pctile':>10} {'d_rand_p95':>11} {'d_len':>7} {'RA>p95?':>8}")
    wins = []
    for F in families:
        pos = (fam_w == F) & (harm_w == 1)
        sel = pos | neg
        y = harm_w[sel]
        if y.min() == y.max():
            continue
        def ev(model, cols):
            return _auc(y, model.predict_proba(np.c_[tuple(c[sel] for c in cols)] if len(cols) > 1
                                                else cols[0][sel][:, None])[:, 1])
        a_int = ev(M_int, [pint_w])
        a_ra  = ev(M_ra,  [pint_w, pra_w])
        a_len = ev(M_len, [pint_w, plen_w])
        a_rnd = np.array([ev(M_rand[k], [pint_w, prand_w[k]]) for k in range(n_rand)])
        dRA, dlen = a_ra - a_int, a_len - a_int
        dnull = a_rnd - a_int
        drnd95 = float(np.percentile(dnull, 95))
        ra_pctile = float((dnull < dRA).mean() * 100)   # where RA sits in the null dist
        win = (dRA > drnd95) and (dRA > dlen) and (dRA > 0)
        wins.append((F, win, dRA, ra_pctile))
        print(f"{F:>20} {int(pos.sum()):>4} {a_int:>8.3f} {dRA:>+7.3f} {ra_pctile:>9.0f}% {drnd95:>+11.3f} {dlen:>+7.3f} "
              f"{'YES' if win else 'no':>8}")

    won = [F for F, w, _, _ in wins if w]
    beat_center = [F for F, _, dra, pc in wins if pc > 50]
    print(f"{'='*100}")
    print(f"RA beats null p95 on: {won or 'NONE'}  ({len(won)}/{len(wins)})")
    print(f"RA above null CENTER (>p50) on: {len(beat_center)}/{len(wins)} families")
    print("READ: p95 is the strict bar; RA_pctile>>50 across families = weak-but-non-null (1.7B);")
    print("RA_pctile ~50 or below = genuinely null/anti (4B). Do NOT aggregate — 4B@L33 is trustworthy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
