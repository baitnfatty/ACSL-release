"""Run baselines and ACSL on identical prompts; emit a paired results table.

A "system" is any callable ``prompt:str -> response:str``. The harness is
generation-backend agnostic: stub systems (e.g. refuse-all) work in tests; real
systems wrap ``model.generate`` + an output gate, or the ACSL loop/policy.

Every safety number (ASR) is reported next to over-refusal on the SAME prompts,
with bootstrap CIs, plus a paired diff of harmful-compliance vs a baseline.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

from .metrics import asr, over_refusal, bootstrap_ci, paired_diff


def _compliance_flags(responses, judge):
    return [1.0 if judge(r) == "compliance" else 0.0 for r in responses]


def _refusal_flags(responses, judge):
    return [1.0 if judge(r) == "refusal" else 0.0 for r in responses]


def run_eval_table(
    systems: Mapping[str, Callable[[str], str]],
    harmful_prompts: Sequence[str],
    benign_prompts: Sequence[str],
    judge: Callable[[str], str],
    baseline: str | None = None,
    n_boot: int = 10000,
    seed: int = 0,
) -> dict:
    """Evaluate every system on the same prompts; return a results dict.

    Returns ``{'rows': [...per system...], 'baseline': name}``. Each row has
    asr + over_refusal with 95% CIs and (vs baseline) a paired ASR diff. The
    baseline defaults to the first system listed.
    """
    names = list(systems.keys())
    if not names:
        raise ValueError("no systems provided")
    baseline = baseline or names[0]
    if baseline not in systems:
        raise ValueError(f"baseline {baseline!r} not among systems {names}")

    # Generate once per system on the shared prompt sets.
    gen = {
        name: {
            "harm": [sys(p) for p in harmful_prompts],
            "benign": [sys(p) for p in benign_prompts],
        }
        for name, sys in systems.items()
    }

    base_comply = _compliance_flags(gen[baseline]["harm"], judge)

    rows = []
    for name in names:
        harm_resp = gen[name]["harm"]
        benign_resp = gen[name]["benign"]
        comply = _compliance_flags(harm_resp, judge)
        refuse = _refusal_flags(benign_resp, judge)

        row = {
            "system": name,
            "asr": asr(harm_resp, judge),
            "asr_ci": bootstrap_ci(comply, n_boot=n_boot, seed=seed),
            "over_refusal": over_refusal(benign_resp, judge),
            "over_refusal_ci": bootstrap_ci(refuse, n_boot=n_boot, seed=seed),
            "n_harm": len(harm_resp),
            "n_benign": len(benign_resp),
        }
        if name != baseline:
            # diff = this system's compliance - baseline's (negative == safer)
            row["asr_vs_baseline"] = paired_diff(comply, base_comply, n_boot=n_boot, seed=seed)
        rows.append(row)

    return {"rows": rows, "baseline": baseline}


def format_table(result: dict) -> str:
    """Render the result dict as a compact text table."""
    lines = [
        f"baseline: {result['baseline']}",
        f"{'system':<22} {'ASR':>7} {'ASR 95% CI':>20} {'OverRef':>8} {'OverRef 95% CI':>20}",
        "-" * 80,
    ]
    for r in result["rows"]:
        ci = r["asr_ci"]
        orci = r["over_refusal_ci"]
        lines.append(
            f"{r['system']:<22} {r['asr']:>7.3f} "
            f"[{ci[0]:>6.3f},{ci[1]:>6.3f}]   "
            f"{r['over_refusal']:>8.3f} "
            f"[{orci[0]:>6.3f},{orci[1]:>6.3f}]"
        )
    return "\n".join(lines)
