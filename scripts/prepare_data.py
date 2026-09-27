#!/usr/bin/env python
"""prepare_data.py - fetch public safety benchmarks into the ACSL jsonl schema.

Downloads established, widely-cited public research benchmarks directly from
their canonical source files (no auth, no content generation; the harness only
classifies model responses as refusal vs compliance):

  data/harm.jsonl              AdvBench harmful instructions (llm-attacks repo)  label 1
  data/harmless.jsonl          Alpaca benign instructions (stanford_alpaca)      label 0
  data/benign_sensitive.jsonl  XSTest 'safe' prompts (over-refusal probe)        label 0

After running, point configs/*.yaml data.* at the printed paths.

    python scripts/prepare_data.py --n-harm 256 --n-harmless 256 --n-benign 250
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import urllib.request

ADVBENCH = [
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv",
]
ALPACA = [
    "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json",
]
XSTEST = [
    "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv",
    "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_v2_prompts.csv",
    "https://raw.githubusercontent.com/paul-rottger/xstest/main/prompts.csv",
]
# Public in-the-wild jailbreak TEMPLATES ("Do Anything Now", Shen et al.). We only
# *reference* these existing public artifacts at runtime to wrap our harm prompts;
# nothing is bundled in the repo and no novel jailbreak is generated.
JAILBREAK_TEMPLATES = [
    "https://raw.githubusercontent.com/verazuo/jailbreak_llms/main/data/prompts/jailbreak_prompts_2023_12_25.csv",
    "https://raw.githubusercontent.com/verazuo/jailbreak_llms/main/data/prompts/jailbreak_prompts_2023_05_07.csv",
]


def fetch(url: str, timeout: int = 180) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "acsl-prepare/1.0"})
    return urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", "replace")


def fetch_first(urls):
    last = None
    for u in urls:
        try:
            return fetch(u), u
        except Exception as e:  # try next candidate
            last = e
    raise RuntimeError(f"all sources failed; last error: {last}")


def write_jsonl(path, prompts, label):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, p in enumerate(prompts):
            f.write(json.dumps({"id": f"{label}-{i}", "prompt": p, "label": label}) + "\n")
    return len(prompts)


def build_wrapped(harm_prompts, n_templates: int, rng, max_template_chars: int = 1400):
    """Wrap harm prompts in EXISTING public jailbreak templates (surface-form change,
    same intent). Returns (wrapped_prompts, source_url)."""
    text, src = fetch_first(JAILBREAK_TEMPLATES)
    rows = list(csv.DictReader(io.StringIO(text)))
    templates = [(r.get("prompt") or "").strip() for r in rows]
    # keep reasonable-length, de-duplicated templates
    seen, pool = set(), []
    for t in templates:
        if t and len(t) <= max_template_chars and t not in seen:
            seen.add(t)
            pool.append(t)
    rng.shuffle(pool)
    pool = pool[: max(1, n_templates)]
    wrapped = [f"{pool[i % len(pool)]}\n\n{goal}" for i, goal in enumerate(harm_prompts)]
    return wrapped, src


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch public safety benchmarks -> jsonl")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--n-harm", type=int, default=256)
    ap.add_argument("--n-harmless", type=int, default=256)
    ap.add_argument("--n-benign", type=int, default=250)
    ap.add_argument("--with-wrapped", action="store_true",
                    help="also build data/wrapped.jsonl: harm prompts wrapped in public jailbreak templates")
    ap.add_argument("--n-templates", type=int, default=8, help="distinct jailbreak templates to cycle")
    ap.add_argument("--hard", action="store_true",
                    help="also build de-saturated hard-negative train set + held-out benign eval (XSTest)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # --- harmful: AdvBench harmful_behaviors.csv (column 'goal') -----------
    text, src = fetch_first(ADVBENCH)
    rows = list(csv.DictReader(io.StringIO(text)))
    harm = [(r.get("goal") or r.get("prompt") or "").strip() for r in rows]
    harm = [h for h in harm if h][: args.n_harm]
    print(f"[prepare_data] harm <- {src}  ({len(harm)})")

    # --- harmless: Alpaca instructions with no 'input' --------------------
    text, src = fetch_first(ALPACA)
    alp = json.loads(text)
    pool = [d["instruction"].strip() for d in alp
            if d.get("instruction") and not (d.get("input") or "").strip()]
    rng.shuffle(pool)
    harmless = pool[: args.n_harmless]
    print(f"[prepare_data] harmless <- {src}  ({len(harmless)})")

    # --- XSTest: 'safe' (benign-sensitive) and 'unsafe' (surface-matched harm) -
    benign, xs_safe_all, xs_unsafe_all = [], [], []
    try:
        text, src = fetch_first(XSTEST)
        rows = list(csv.DictReader(io.StringIO(text)))

        def is_safe(r):
            lab = str(r.get("label", "")).lower()
            if lab:
                return lab.startswith("safe")
            t = str(r.get("type", "")).lower()
            return bool(t) and not t.startswith("contrast")

        for r in rows:
            p = (r.get("prompt") or "").strip()
            if not p:
                continue
            (xs_safe_all if is_safe(r) else xs_unsafe_all).append(p)
        benign = xs_safe_all[: args.n_benign]
        print(f"[prepare_data] XSTest <- {src}  safe={len(xs_safe_all)} unsafe={len(xs_unsafe_all)}")
    except Exception as e:
        print(f"[prepare_data] WARNING: XSTest fetch failed ({e}); skipping benign_sensitive for now")

    nh = write_jsonl(os.path.join(args.outdir, "harm.jsonl"), harm, 1)
    nl = write_jsonl(os.path.join(args.outdir, "harmless.jsonl"), harmless, 0)
    nb = write_jsonl(os.path.join(args.outdir, "benign_sensitive.jsonl"), benign, 0) if benign else 0

    # --- wrapped: harm prompts in EXISTING public jailbreak templates ------
    nw = 0
    if args.with_wrapped:
        try:
            wrapped, src = build_wrapped(harm, args.n_templates, rng)
            nw = write_jsonl(os.path.join(args.outdir, "wrapped.jsonl"), wrapped, 1)
            print(f"[prepare_data] wrapped <- {src}  ({nw}; {args.n_templates} templates)")
        except Exception as e:
            print(f"[prepare_data] WARNING: wrapped build failed ({e}); skipping")

    # --- de-saturated hard set: surface-matched harm/harmless, held-out eval -
    if args.hard:
        if not xs_safe_all or not xs_unsafe_all:
            print("[prepare_data] WARNING: --hard needs XSTest safe+unsafe; skipping")
        else:
            safe = list(xs_safe_all)
            rng.shuffle(safe)
            mid = len(safe) // 2
            safe_train, safe_eval = safe[:mid], safe[mid:]   # split to avoid leakage
            harm_hard = harm + xs_unsafe_all                 # AdvBench + XSTest-unsafe
            harmless_hard = harmless + safe_train            # Alpaca + hard benign negatives
            rng.shuffle(harm_hard)
            rng.shuffle(harmless_hard)
            write_jsonl(os.path.join(args.outdir, "harm_hard.jsonl"), harm_hard, 1)
            write_jsonl(os.path.join(args.outdir, "harmless_hard.jsonl"), harmless_hard, 0)
            write_jsonl(os.path.join(args.outdir, "benign_eval.jsonl"), safe_eval, 0)
            print(f"[prepare_data] HARD: harm_hard={len(harm_hard)} (advbench+xstest-unsafe)  "
                  f"harmless_hard={len(harmless_hard)} (alpaca+xstest-safe-half)  "
                  f"benign_eval={len(safe_eval)} (held-out xstest-safe)")

    print(f"\n[prepare_data] wrote harm={nh}  harmless={nl}  benign_sensitive={nb}  wrapped={nw}  -> {args.outdir}/")
    print("[prepare_data] config data paths:")
    print(f"    harm_path:             {args.outdir}/harm.jsonl")
    print(f"    harmless_path:         {args.outdir}/harmless.jsonl")
    if nb:
        print(f"    benign_sensitive_path: {args.outdir}/benign_sensitive.jsonl")
    if nw:
        print(f"    wrapped_path:          {args.outdir}/wrapped.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
