#!/usr/bin/env python
"""13_gate_validation — trust-the-instrument suite for the intent/RA probe axes.

Run BEFORE any per-family rescue claim. Per axis, per operating layer, on cached
activations (the same caches the server reads for its directions):

  1. Polarity / known-answer — held-out blatant-harm vs blatant-benign raw signed
     scores. Harm must score high, benign low; report inversions.
  2. Positive control — a trivially lexical label (token presence) must be decodable
     at ~1.0 AUROC, else extraction is broken.
  3. Shuffle test — permute labels, refit, AUROC must collapse to ~0.5.
  4. Degenerate-shortcut check — regress the axis score on prompt length, token count,
     punctuation, wrapper-presence; report variance explained (R^2).
  5. Cross-family transfer — fit on train jailbreak families, test on HELD-OUT families.

    python scripts/13_gate_validation.py --config configs/v3.yaml
    python scripts/13_gate_validation.py --config configs/v3_4b.yaml
"""
from __future__ import annotations
import argparse, importlib.util, json, os, re, string, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from acsl.extract import load_cached
from acsl.norm import load_norm_stats
from acsl.config import load_config

# single-source the RA labeler from script 11 (module name starts with a digit)
_spec = importlib.util.spec_from_file_location(
    "ra11", os.path.join(os.path.dirname(os.path.abspath(__file__)), "11_request_actionability.py"))
_ra11 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_ra11)
label_request_actionability = _ra11.label_request_actionability

# operating layers per model (STATUS "Decisions made")
OP_LAYERS = {"cache_v3": {"intent": 9, "ra": 9},
             "cache_v3_4b": {"intent": 33, "ra": 20}}


def _dom(Xn, lab):
    d = Xn[lab == 1].mean(0) - Xn[lab == 0].mean(0)
    n = np.linalg.norm(d)
    return d / n if n > 0 else d

def _auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s)

def _cv_auroc(Xn, lab, seed=0):
    """5-fold: refit diff-of-means direction on train, project test, pool OOF scores."""
    from sklearn.model_selection import StratifiedKFold
    if lab.sum() < 5 or (1 - lab).sum() < 5:
        return float("nan")
    oof = np.zeros(len(lab))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xn, lab):
        u = _dom(Xn[tr], lab[tr])
        oof[te] = Xn[te] @ u
    return _auroc(lab, oof)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    cache_dir = cfg.get_path("data.cache_dir")
    key = "cache_v3_4b" if "cache_v3_4b" in cache_dir else "cache_v3"
    ns = load_norm_stats(os.path.join(cache_dir, "norm_stats.npz"))

    rec = {}
    for p in ["data/v3_harmful.jsonl", "data/v3_harmless.jsonl", "data/v3_wrapped.jsonl"]:
        for line in open(p):
            r = json.loads(line); rec[r["id"]] = r

    def load(split, L):
        X, _, ids = load_cached(cache_dir, split, L)
        mu = ns[L]["mu"]; sig = np.clip(ns[L]["sigma"], 1e-12, None)
        ids = [str(i) for i in ids]
        return (X - mu) / sig, ids

    def harm(ids):  return np.array([int(rec[i].get("intent", 0) >= 1) if i in rec else 0 for i in ids])
    def ra(ids):    return np.array([label_request_actionability(rec[i]["prompt"]) if i in rec else 0 for i in ids])
    def prompts(ids): return [rec[i]["prompt"] if i in rec else "" for i in ids]

    print(f"\n{'='*80}\nGATE VALIDATION — {args.config}  cache={cache_dir}\n{'='*80}")

    for axis in ("intent", "ra"):
        L = OP_LAYERS[key][axis]
        Xc, idc = load("clean", L)
        lab = harm(idc) if axis == "intent" else ra(idc)
        axis_name = "Intent (harm label)" if axis == "intent" else "Request-Actionability (RA label)"
        print(f"\n{'-'*80}\nAXIS: {axis_name}  @ operating layer L{L}\n{'-'*80}")

        # ---- Test 1: polarity / known-answer (held-out) ----
        from sklearn.model_selection import train_test_split
        sev = np.array([int(rec[i].get("severity", 0)) if i in rec else 0 for i in idc])
        idx = np.arange(len(lab))
        tr_i, te_i = train_test_split(idx, test_size=0.3, stratify=lab, random_state=0)
        u_tr = _dom(Xc[tr_i], lab[tr_i])                  # fit on train only
        sc = Xc @ u_tr
        te_pos, te_neg = te_i[lab[te_i] == 1], te_i[lab[te_i] == 0]
        if axis == "intent":                              # prefer BLATANT within held-out test
            bp = te_i[(lab[te_i] == 1) & (sev[te_i] == 2)]
            bn = te_i[(lab[te_i] == 0) & (sev[te_i] == 0)]
            if len(bp) >= 10 and len(bn) >= 10:
                te_pos, te_neg = bp, bn
        pos_mean, neg_mean = float(sc[te_pos].mean()), float(sc[te_neg].mean())
        inverted = pos_mean <= neg_mean
        print(f"[1] POLARITY (held-out, n_pos={len(te_pos)} n_neg={len(te_neg)}): "
              f"pos_mean={pos_mean:+.3f}  neg_mean={neg_mean:+.3f}  gap={pos_mean-neg_mean:+.3f}  "
              f"{'*** INVERTED ***' if inverted else 'OK (pos>neg)'}")

        # ---- Test 2: positive control (lexical token presence, full-dim logistic ~1.0) ----
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict
        cands = ["you", "how", "my", "what", "the", "?"]
        pl = prompts(idc)
        best = None
        for tok in cands:
            tl = np.array([1 if tok in s.lower() else 0 for s in pl])
            bal = min(tl.mean(), 1 - tl.mean())
            if best is None or bal > best[1]:
                best = (tok, bal, tl)
        tok, bal, tl = best
        proba = cross_val_predict(LogisticRegression(max_iter=2000), Xc, tl, cv=5,
                                  method="predict_proba")[:, 1]
        pc_auroc = _auroc(tl, proba)
        print(f"[2] POSITIVE CONTROL: token '{tok}' present (bal={bal:.2f}) full-dim CV AUROC={pc_auroc:.3f}  "
              f"{'OK (~1.0)' if pc_auroc > 0.95 else '*** EXTRACTION SUSPECT ***'}")

        # ---- Test 3: shuffle test (permute labels -> AUROC ~0.5) ----
        rng = np.random.RandomState(0)
        shuf = [ _cv_auroc(Xc, rng.permutation(lab), seed=s) for s in range(5) ]
        real = _cv_auroc(Xc, lab)
        print(f"[3] SHUFFLE: real CV AUROC={real:.3f}  shuffled mean={np.mean(shuf):.3f} "
              f"(range {min(shuf):.3f}-{max(shuf):.3f})  {'OK (~0.5)' if np.mean(shuf) < 0.6 else '*** LEAK ***'}")

        # ---- Test 4: degenerate-shortcut (variance of score explained by surface) ----
        Xw, idw = load("wrapped", L)
        Xall = np.vstack([Xc, Xw]); ids_all = idc + idw
        u_all = _dom(Xc, lab)                              # axis direction (fit on clean)
        score = Xall @ u_all
        pl = prompts(ids_all)
        feats = {
            "char_len":   np.array([len(s) for s in pl], float),
            "token_cnt":  np.array([len(s.split()) for s in pl], float),
            "punct_cnt":  np.array([sum(c in string.punctuation for c in s) for s in pl], float),
            "wrapper_present": np.array([0]*len(idc) + [1]*len(idw), float),  # clean=0, wrapped=1
        }
        print(f"[4] SHORTCUT R^2 (score ~ surface feature, clean+wrapped n={len(pl)}):")
        for name, f in feats.items():
            r2 = float(np.corrcoef(score, f)[0, 1] ** 2) if f.std() > 0 else float("nan")
            flag = " <-- high" if r2 > 0.25 else ""
            print(f"      {name:16s} R^2={r2:.3f}{flag}")

        # ---- Test 5: cross-family transfer (harm dir from TRAIN families vs clean benign,
        #      tested on HELD-OUT families). Wrapper families are all-harmful, so the
        #      negative class is clean-benign (shared). Tests: does a harm detector learned
        #      from some jailbreak families detect harm under UNSEEN jailbreak families? ----
        fam_split = np.array([rec[i].get("family_split", "?") if i in rec else "?" for i in idw])
        yw = harm(idw)
        pos_tr = (fam_split == "train") & (yw == 1)
        pos_ho = (fam_split == "holdout") & (yw == 1)
        neg = harm(idc) == 0                               # clean benign as shared negative
        if pos_tr.sum() > 10 and pos_ho.sum() > 10 and neg.sum() > 10:
            Xpos_tr, Xpos_ho, Xneg = Xw[pos_tr], Xw[pos_ho], Xc[neg]
            u_fam = _dom(np.vstack([Xpos_tr, Xneg]),
                         np.r_[np.ones(len(Xpos_tr)), np.zeros(len(Xneg))].astype(int))
            def auc(Xp):
                s = np.r_[Xp @ u_fam, Xneg @ u_fam]
                y = np.r_[np.ones(len(Xp)), np.zeros(len(Xneg))]
                return _auroc(y, s)
            auc_tr, auc_ho = auc(Xpos_tr), auc(Xpos_ho)
            print(f"[5] CROSS-FAMILY (harmful-wrapped vs clean-benign): "
                  f"train-family AUROC={auc_tr:.3f}  HELD-OUT-family AUROC={auc_ho:.3f}  "
                  f"transfer gap={auc_tr-auc_ho:+.3f}  {'*** POOR TRANSFER ***' if auc_tr-auc_ho > 0.1 else 'OK'}")
        else:
            print("[5] CROSS-FAMILY: insufficient examples — skipped")

    print(f"\n{'='*80}\nDONE. Nothing committed.\n{'='*80}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
