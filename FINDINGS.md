# ACSL — Findings Report (interim)

**Date:** 2026-06-28
**Model:** `Qwen/Qwen2.5-1.5B-Instruct` (frozen, eager attention)
**Status:** Headline numbers from the **n=200** paired run (complete), incl. the
ROC. Probe data not yet de-saturated (§6.1) — that pass is the next run.

> Scope reminder: this is defensive safety-evaluation tooling. It consumes
> **existing public** safety benchmarks and classifies model responses as
> refusal vs compliance. No novel jailbreaks or harmful content were authored.

---

## 1. TL;DR

On `Qwen2.5-1.5B-Instruct`, with harmful prompts wrapped in **existing public**
jailbreak templates:

- The **output** refusal gate is fooled **~22%** of the time on wrapped-harm
  (baseline ASR 0.000 clean → **0.219 wrapped**).
- The **internal** signal (a SecurityHead read at attention layer **L=9**)
  catches **all** of those at matched over-refusal (**ACSL ASR 0.000**).
- **ASR reduction ≈ 0.219**, 95% CIs disjoint, at the same over-refusal (0.438).

This is the mechanism the architecture predicted — *and it was obtained with
public attacks only*. **But the saturated probe data made it look better than it
is**: see the de-saturated re-run (§5b), which is the real test. Net finding —
the internal harm signal is genuinely jailbreak-invariant, **but only at
mid-network layers (≈L13–16); the layer a naive AUROC sweep picks (L20) is
surface-form dependent and fails.** Layer selection must use kill-test
invariance, not clean AUROC. And the residual-failure decomposition (§5c) pins
down *where* it stops: gate-misses are specific jailbreak templates **suppressing
an otherwise-present signal** (85% suppressed-by-wrapper, not weak-clean),
concentrated in a couple of persona-override templates — a **hardenable** failure
mode, not a probe ceiling.

---

## 2. Runtime / hardware (how we got the GPU working)

The target card is an **AMD RX 6900 XT (RDNA2, gfx1030)**.

- **ROCm-on-WSL (RHEL 10) — dead end.** We built AMD's `librocdxg` shim from
  source, registered the device id (`0x73BF`), installed ROCm 7.2.4 userspace,
  and the GPU **enumerated** over `/dev/dxg` (`rocminfo`, `torch.cuda` device
  name all worked). But **compute submission hangs** (`rptr` never advances):
  AMD's ROCm-on-WSL only schedules compute for RDNA3/RDNA4. RDNA2 enumerates but
  won't execute. Not a config bug — a driver-level GPU-generation gate.
- **DirectML (native Windows) — works.** `torch-directml` runs PyTorch on any
  DX12 GPU, RDNA2 included. The frozen Qwen2.5-1.5B forward + hooks + the whole
  pipeline run on the 6900 XT via the `privateuseone` device. fp32 (DirectML
  fp16 coverage is incomplete). This is the environment all results below use.

---

## 3. Setup

- **Frozen base**, params `requires_grad_(False)`, eager attention, hooked at the
  attention sublayer output `o_L`.
- **Internal SecurityHead:** a small classifier trained only on cached `o_L`
  (last-token, chat-templated → read at the true end-of-instruction position).
  Base never backprops.
- **Data (public benchmarks, fetched at runtime, not bundled):**
  - harm: **AdvBench** harmful behaviors (256)
  - harmless: **Alpaca** instructions (256)
  - benign-sensitive: **XSTest** "safe" prompts (250) — over-refusal probe
  - wrapped: AdvBench prompts in **public jailbreak templates** ("Do Anything
    Now", verazuo/jailbreak_llms), 8 templates (256)

---

## 4. Pipeline results

| step | what | result |
|---|---|---|
| 01 extract | `o_L` across layers 8–20, ~1000 fwd passes on DirectML | clean=512, wrapped=512, benign=250 |
| 02 sweep | head per layer, pick best by val AUROC | **AUROC ≈ 1.0 at *every* layer** (best L=9) |
| 03 faithfulness | grounding / internality / spuriousness at L9 | inconclusive (see §6); risk **not** explained by activation magnitude (corr −0.01) |
| 06 kill test | head AUROC clean vs jailbreak-wrapped | clean 0.9996, wrapped 0.9985, **Δ=0.001 ⇒ "VIABLE"** |
| 07 paired | output gate vs internal gate (n=64) | **headline, §5** |

Two of these carry health warnings:
- **02 is saturated.** AUROC=1.0 at all 13 layers means AdvBench-vs-Alpaca is
  *trivially* separable (different topic/format), so the sweep can't pick a real
  L and the probe may be reading surface form. L=9 is the first to hit 1.0.
- **06 is necessary-not-sufficient.** Δ≈0 only means something if the wrapper
  actually changes model behavior — which §5 is what tests.

---

## 5. Headline: output gate vs internal gate (n=200)

```
                 output complies   mean internal risk
clean_harm       0.005             +6.34   (model refuses ~all clean AdvBench)
wrapped_harm     0.175             +5.13   (17.5% jailbroken at the OUTPUT)
benign (XSTest)  0.485 (OR 0.515)  -5.00

Paired, at matched over-refusal = 0.515:
  baseline output-gate ASR (wrapped) = 0.175   95% CI [0.125, 0.230]
  ACSL internal-gate  ASR (wrapped)  = 0.000   95% CI [0.000, 0.000]

ROC — ACSL ASR on wrapped vs over-refusal budget (baseline ref = 0.175):
   2% over-refusal -> ACSL ASR 0.010
   5% over-refusal -> ACSL ASR 0.000
  10% over-refusal -> ACSL ASR 0.000
  20% over-refusal -> ACSL ASR 0.000
```

**Reading:** the wrapper flips 17.5% of harmful prompts from refusal to compliance
at the output; the internal harm signal at L9 fires on essentially all of them.
Crucially the **ROC** shows this is not an artifact of a lax operating point — the
internal gate drives wrapped ASR 0.175 → ~0 across the whole over-refusal range,
catching all but ~1 jailbroken prompt even at a **2%** over-refusal budget. The
internal signal separates wrapped-harm (mean risk +5.1) from benign-sensitive
XSTest (−5.0) by a wide margin. This addresses caveat §6.2.

---

## 5b. De-saturated re-run (the real test)

`configs/hard.yaml`: harm = AdvBench + XSTest-**unsafe**; harmless = Alpaca +
half of XSTest-**safe** (hard, benign-but-sensitive negatives); over-refusal eval
= held-out other half of XSTest-safe. This forces the probe to read **intent**,
not topic/format.

**Layer sweep is now informative** (no longer pinned at 1.0): val AUROC ranges
0.94 (L9) → 0.997 (L20); naive "best clean AUROC" = **L20**.

**Kill test swept across layers** (clean vs jailbreak-wrapped AUROC; trained head
per layer):

| layer | clean | wrapped | delta | verdict |
|------:|------:|--------:|------:|---------|
| 8  | 0.947 | 0.637 | +0.310 | killed (too early) |
| 9  | 0.939 | 0.951 | −0.012 | invariant |
| 13 | 0.988 | 0.994 | −0.006 | invariant |
| **15** | **0.997** | **0.992** | **+0.005** | **invariant + discriminative** |
| 16 | 0.996 | 0.983 | +0.013 | invariant |
| 18 | 0.996 | 0.973 | +0.023 | invariant |
| 19 | 0.993 | 0.901 | +0.092 | killed |
| 20 | 0.997 | 0.851 | +0.146 | **killed** (what 02 picked) |

**Key results:**
1. On de-saturated data the **L20** head (picked by clean AUROC) **fails** the
   kill test — delta 0.147. The §5 saturated "VIABLE" was partly an artifact of a
   probe reading surface form.
2. But **mid-layers (13–16) are both discriminative and jailbreak-invariant**.
   **L15: clean 0.997, wrapped 0.992, delta +0.005** — the harm signal survives
   wrapping even on intent-focused data.
3. **Methodological fix (implemented):** `02_sweep_train_head` now selects the
   most discriminative layer **among those passing the kill-test invariance
   threshold** (clean−wrapped delta ≤ `decision.kill_delta_max`), falling back to
   clean AUROC only when no wrapped split exists. On the de-saturated data it now
   auto-picks **L15** (delta 0.005) instead of L20, and the end-to-end pipeline
   (01→02→06) terminates **VIABLE**.

**Capstone — paired ASR at the invariance-selected L15, de-saturated data,
held-out XSTest-safe benign (n=125, no leakage):**

```
baseline output ASR: clean 0.020 -> wrapped 0.175   (jailbreak works)
ACSL ASR at matched over-refusal (0.536) = 0.000

ROC (ACSL ASR on wrapped vs over-refusal budget; baseline ref 0.175):
   2% over-refusal -> 0.135     5% -> 0.045     10% -> 0.010     20% -> 0.005
```

Honest read: the internal gate **does** catch jailbroken-harm the output misses,
but the effect is **operating-point-dependent and modest at tight budgets** — at
a 2% over-refusal budget it barely beats baseline (0.135 vs 0.175); it earns its
keep at 5–10% over-refusal (0.045 / 0.010, a 75–95% ASR cut). Contrast the
saturated L9 run (2%→0.01): **the easy data overstated the effect**; every rigor
step (de-saturate → invariance-select → held-out benign → ROC) deflated it toward
this honest, modest-positive result. One model, one attack family — not general.

---

## 5c. Residual-failure decomposition — where the mechanism stops

L15, n=256 wrapped-harm. For each intent we measured internal risk **clean vs
wrapped**, output **compliance**, and the **jailbreak template**.

```
risk_clean mean +10.44  ->  risk_wrapped mean +5.74   (wrapper weakens signal ~4.7, stays strongly positive)
output complied = 0.164

budget 5%  (thr +4.08): gate-miss 30.5%  residual ASR 4.7%  | 78 misses: 66 suppressed-by-wrapper, 12 weak-clean
budget 10% (thr +2.15): gate-miss 10.5%  residual ASR 1.2%  | 27 misses: 25 suppressed-by-wrapper,  2 weak-clean
```

per-template gate-miss / residual-ASR / mean-suppression (5% budget):

| tmpl | miss/32 | residual | suppression |
|-----:|--------:|---------:|------------:|
| 2 | 22 | 1 | +7.15 |
| 4 | 15 | 0 | +6.67 |
| 0 | 12 | 0 | +5.94 |
| 1 |  9 | 1 | +2.75 |
| 5 |  6 | 0 | +3.92 |
| 6 |  6 | 5 | +3.96 |
| 3 |  5 | 2 | +4.73 |
| 7 |  3 | 3 | +2.52 |

**Three findings:**
1. **Failures are attack-induced suppression, not probe blindness.** 85% of
   gate-misses (66/78 at 5%) were clearly flagged when clean and only pushed below
   threshold by the wrapper; just 12 were weak even clean. The probe *can*
   represent these intents — specific wrappers hide them.
2. **Suppression is template-specific and concentrated.** Template 2 misses 22/32
   (mean suppression +7.2 logits); 0 and 4 next. A couple of persona-override
   templates do most of the damage; others barely suppress (or raise) the signal.
3. **The dangerous residual is small and decoupled from suppression.** The
   high-gate-suppression templates (2/0/4) mostly *don't* jailbreak the output, so
   they contribute ~0 residual ASR; the actual residual ASR (12 → 3 cases as the
   budget tightens) comes from output-effective templates (6/7/3). The probe's
   representational failure and the system's operational failure are largely
   disjoint.

**Implication (actionable):** because misses are wrapper-*suppression* of a
present signal (not an absent one), the mechanism is likely **hardenable** — the
suppression may be layer-specific (a multi-layer / ensembled read could recover
what L15 alone loses), and training the head on wrapped examples should help. The
failure is localized to specific surface forms, not a probe ceiling. (`09_residual_decomp.py`.)

---

## 6. Caveats / threats to validity (read before believing §5)

1. **Probe data is saturated.** AUROC=1.0 everywhere ⇒ AdvBench/Alpaca separates
   by topic, not necessarily intent. The clean result may partly reflect surface
   form. *Fix:* matched harm/harmless pairs, or harder negatives.
2. **Operating point is high over-refusal.** Both sit at OR=0.438 on XSTest
   (Qwen2.5-1.5B over-refuses a lot). ACSL hits ASR 0 *there*; the full ROC is
   the real claim, not one point.
3. **Over-refusal accounting is approximate.** We matched the gate's firing rate
   to the baseline's refusal rate; a fully fair version counts ACSL's *total*
   over-refusal (gate-refusals ∪ model-refusals on passed prompts).
4. **Modest n / partial attack.** n=64; baseline CI is wide [0.125, 0.328]; the
   attack is only ~22% effective.
5. **Single model, single attack family.** One small model, one template set.
6. **Wrapped prompts retain the goal verbatim.** The kill test (§4) is weakened
   by the goal being intact at the read position.

---

## 7. What this does and doesn't show

- **Does:** on a real frozen model, an internal mid-layer signal catches harmful
  intent that a public jailbreak hides from the output gate — at matched
  over-refusal — using only existing public attacks.
- **Doesn't (yet):** that the signal reads *intent* rather than surface form
  (needs de-saturated data); that the advantage holds at low over-refusal (needs
  the ROC); that it generalizes across models/attacks.

---

## 8. Next steps

1. **Full ROC + larger n** (in progress, n=200).
2. **De-saturate**: matched harm/harmless or XSTest-unsafe as harm, so AUROC isn't
   pinned at 1.0 and the layer sweep + faithfulness become informative.
3. **Fix over-refusal accounting** to total (gate ∪ model).
4. **Stronger/again attacks** the user supplies, to push baseline ASR up and test
   the gate under harder pressure.
5. **Faithfulness metric fix**: grounding should compare slope (effect size)
   along `u` vs random, not correlation (which saturates for a near-linear head).

---

## 9. Reproducibility

- Env: Windows `.venv` (torch-directml 0.2.5 / torch 2.4.1), `pip install -e .`.
- Data: `python scripts/prepare_data.py --with-wrapped` (fetches public sources).
- Run order: `01_extract → 02_sweep_train_head → 03_faithfulness → 06_kill_test
  → 07_paired_eval`, each with `--config configs/default.yaml` and a run manifest
  under `runs/<script>/manifest.json` (config + git SHA + eval-set hashes).
- Seed: `config.seed`. All numbers above are from layer L=9, head 2-category.
