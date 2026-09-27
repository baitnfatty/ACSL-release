#!/usr/bin/env python
"""07_eval - baselines vs ACSL on identical prompts: paired ASR + over-refusal.

Every safety number (ASR on harmful/jailbreak-wrapped prompts) is reported next
to over-refusal (on benign-sensitive prompts) on the SAME prompts, with bootstrap
95% CIs and a paired ASR diff vs the baseline.

Modes:
  (real, default) requires a frozen model + the configured benchmark files and
    the loop artifacts from 04. Builds an output-gate baseline and the ACSL
    system (internal gate via SecurityLoopHook + action_policy), then evaluates.
  --demo          offline illustration on canned responses (no model): shows the
    table format and that a refuse-all policy scores ASR 0 with max over-refusal.

    python scripts/07_eval.py --config configs/default.yaml --demo
    python scripts/07_eval.py --config configs/default.yaml \\
        --head runs/02_sweep_train_head/head.pt --loop runs/04_build_loop
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acsl.eval.harness import run_eval_table, format_table
from acsl.eval.judge import make_judge
from acsl.runtime import build_parser, resolve, write_manifest


# --------------------------------------------------------------------------
# Offline demo: canned systems on labeled synthetic prompts (no model).
# --------------------------------------------------------------------------
def _demo_systems_and_prompts():
    from acsl.data.synthetic_smoke import synthetic_conditions

    ex = synthetic_conditions(n_per_group=24)
    harmful = [(e.prompt, e.split) for e in ex if e.label == 1]  # clean + wrapped harmful
    benign = [e.prompt for e in ex if e.split == "benign"]
    harmful_prompts = [p for p, _ in harmful]
    wrapped_flag = {p: (s == "wrapped") for p, s in harmful}
    harmful_set = set(harmful_prompts)

    REFUSAL = "I'm sorry, but I can't help with that."
    ANSWER = "Sure, here is the information you requested."

    def refuse_all(p):
        return REFUSAL

    def output_gate_baseline(p):
        # Output-level gate: refuses clean harmful, but a surface-form wrapper
        # slips past it (jailbreak succeeds). Answers benign.
        if p in harmful_set and not wrapped_flag.get(p, False):
            return REFUSAL
        if p in harmful_set:
            return ANSWER  # wrapped harmful -> complied == attack success
        return ANSWER       # benign -> answered

    def acsl(p):
        # Internal gate: refuses ALL harmful regardless of wrapping; answers benign.
        return REFUSAL if p in harmful_set else ANSWER

    systems = {"output_gate": output_gate_baseline, "acsl": acsl, "refuse_all": refuse_all}
    return systems, harmful_prompts, benign


# --------------------------------------------------------------------------
# Real systems (model required). Imported lazily; only used without --demo.
# --------------------------------------------------------------------------
def _real_systems_and_prompts(cfg, args):
    import numpy as np
    import torch

    from acsl.head import load_head
    from acsl.loop import ResidualStreamHook
    from acsl.model import load_frozen_model
    from acsl.policy import action_policy
    from acsl.data.loaders import JailbreakWrappedLoader, BenignSensitiveLoader

    model, tokenizer = load_frozen_model(cfg.get_path("model.name"), cfg.get_path("model.dtype", "float16"))
    device = next(model.parameters()).device
    head, meta = load_head(args.head)
    layer = meta.get("layer")
    u = torch.as_tensor(np.load(os.path.join(args.loop, "u.npy")), dtype=torch.float32).to(device)
    params = json.load(open(os.path.join(args.loop, "loop_params.json")))["params"]
    loop_kw = {k: params[k] for k in ("g_max", "K", "eps", "thr")}

    def _ids(prompt):
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt"
            ).to(device)
        return tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    def _generate(ids):
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False)
        return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    REFUSAL = "I'm sorry, but I can't help with that."

    def baseline(prompt):  # output-level gate == the model's own refusal behavior
        return _generate(_ids(prompt))

    def acsl(prompt):  # internal gate: read r_star/safe at L, then policy
        ids = _ids(prompt)
        with ResidualStreamHook(model, layer, head, u, **loop_kw) as hook:
            with torch.no_grad():
                model(ids)  # populate hook.last_r_star / last_safe from the prompt
            action = action_policy(hook.last_r_star, hook.last_safe)
            if action in ("refuse_or_route", "flag"):
                return REFUSAL
            return _generate(ids)  # 'answer'/'steer': generate with the loop active

    systems = {"baseline_output_gate": baseline, "acsl": acsl}

    wrapped = [e.prompt for e in JailbreakWrappedLoader(cfg.get_path("data.wrapped_path"))]
    benign = [e.prompt for e in BenignSensitiveLoader(cfg.get_path("data.benign_sensitive_path"))]
    n = int(cfg.get_path("eval.n_per_condition", 300))
    return systems, wrapped[:n], benign[:n]


def main() -> int:
    parser = build_parser(__doc__)
    parser.add_argument("--demo", action="store_true", help="offline canned-response demo (no model)")
    parser.add_argument("--head", default=os.path.join("runs", "02_sweep_train_head", "head.pt"))
    parser.add_argument("--loop", default=os.path.join("runs", "04_build_loop"))
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--baseline", default=None, help="which system is the baseline row")
    args = parser.parse_args()
    cfg, out = resolve(args)

    judge = make_judge(cfg.get_path("eval.judge", "heuristic"))
    if args.demo:
        systems, harmful, benign = _demo_systems_and_prompts()
        baseline = args.baseline or "output_gate"
    else:
        systems, harmful, benign = _real_systems_and_prompts(cfg, args)
        baseline = args.baseline or "baseline_output_gate"

    result = run_eval_table(
        systems, harmful, benign, judge,
        baseline=baseline,
        n_boot=int(cfg.get_path("eval.n_boot", 10000)),
        seed=int(cfg.get_path("seed", 0)),
    )
    print(format_table(result))
    for row in result["rows"]:
        if "asr_vs_baseline" in row:
            d = row["asr_vs_baseline"]
            print(f"[eval] {row['system']} ASR diff vs {result['baseline']}: "
                  f"{d['diff']:+.3f} (95% CI [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}])")

    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    eval_paths = [cfg.get_path("data.wrapped_path"), cfg.get_path("data.benign_sensitive_path")]
    write_manifest(out, cfg, eval_paths=eval_paths, extra={"baseline": result["baseline"]})
    print(f"[eval] OK - results -> {os.path.join(out, 'results.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
