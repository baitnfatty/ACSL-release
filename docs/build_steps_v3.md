# ACSL — Step-by-Step Build & Test (reasoning base, RHEL + ROCm)

Staged. Each step has a **gate** — a result that tells you whether to proceed or stop. Do not skip a gate to get to the next stage; the gates are the experiment.

---

## Step 0 — Environment (once)

- RHEL 9.6 or 10.1 (pinned point release), ROCm 7.2.x, add to `render`/`video`, reboot.
- `rocminfo | grep gfx` → gfx1030. `pip install torch --index-url .../rocm6.2` (or rocm7.x). `torch.cuda.is_available()` → True.
- Load base with `attn_implementation="eager"` (so the attention sublayer is hookable), `torch_dtype=float16`, all params frozen.

## Step 1 — Pick the base and clean house

- Base: **Qwen3-1.7B** (or 4B if VRAM allows). Reason: same family as your harness, and `enable_thinking` toggle gives a free reasoning-on/off ablation later.
- **Delete:** AdvBench-safe-only negatives, the old single-axis head, all Qwen2.5 result files.
- **Keep:** hooks, projection code, metrics, bootstrap CIs, held-out-wrapper logic, kill-test script.
- Run with `enable_thinking=False` for Stages 0–2 (read at prompt encoding, before any trace). Thinking comes back in Step 7.

## Step 2 — Build the data (the load-bearing part)

Schema: `{text, intent 0|1, severity 0|1|2, wrapper_family, split}`. Label by **intent**, never behavior.

- malicious: AdvBench harmful + your curated harmful-that-passes + working wrappers (tag each with a `wrapper_family`).
- benign: **hard** negatives — XSTest-style "looks unsafe but isn't" + defensive security text (CVE writeups, ATT&CK, "explain a buffer overflow," detection rules). Allow **high severity at intent 0**.
- **Split wrapper families into train vs heldout now.** Some families never touch training — they're your invariance test set. This firewall is what keeps the result honest.

**Gate:** if your benign set is mostly easy/general (poems, trivia), stop and fix it. Soft negatives → inflated AUROC that measures topic, not intent.

## Step 3 — Stage 0: find the invariant band  *(the eval you've been deferring)*

```
python scripts/extract.py     # cache h_ℓ at last prompt token, all layers
python scripts/kill_test.py   # per-layer auroc_clean vs auroc_wrapped (HELD-OUT families), per axis
```
- For each axis, plot `auroc_clean[ℓ]` and `auroc_wrapped[ℓ]` across all layers. `delta[ℓ] = clean − wrapped`.
- `B = { ℓ : delta ≤ τ for BOTH intent and severity, AND clean high }`.

**Gate:**
- band exists for both axes → continue, record `B`.
- no band (every layer's wrapped AUROC collapses, or clean never separates) → two-axis ACSL doesn't hold on this model. **Stop.** That's a real (negative) result — write it up.
- expect `B` ≈ 3–5 layers. If you're tempted by 6–9, the curve has to justify it; don't pick the count, read it off the plot.

## Step 4 — Stage 1: test the interaction  *(go/no-go for the suppression head)*

```
python scripts/divergence.py  # div = severity_proj − intent_proj across B
```
- compute `div` for three sets: clean-harmful, wrapped-harmful, benign.
- plot the three distributions; compute separability (AUROC of `div` for wrapped-harmful vs benign).

**Gate:**
- wrapped-harmful separable from benign on `div` → the suppression signature exists, build Stage 2.
- they overlap → **do not build the suppression head.** Nothing to detect. Keep two-axis + trajectory and move on.

## Step 5 — Stage 2: suppression detector  *(only if Step 4 passed)*

```
python scripts/train_suppression.py   # head: [i_B,s_B,Δi,Δs,Δi−Δs] -> wrapped vs genuine
```
- train on **training** wrapper families only.
- evaluate on **held-out** families + benign.
- compose into gate: `suppressed → treat intent as HIGH`.

**Gate:** suppression head must improve the joint metric (Step 6) over two-axis-without-it on **held-out** families. If it only helps on trained families, it memorized wrappers — drop it.

## Step 6 — Evaluate (the number that matters)

```
python scripts/paired_eval.py   # baselines vs ACSL, identical prompts, n≥300/condition
```
Report, paired, with bootstrap 95% CIs:
- **ASR** (held-out wrapped-harmful that the gate lets through) **and over-refusal** (on HARD benign) — together, always.
- baselines: native refusal · output classifier (Llama-Guard-style) · single-layer single-axis · random-direction steering (sanity).
- ablations: one-axis vs two-axis · aggregate vs trajectory(+deltas) · ±suppression head.

**Gate (the honest decision rule, pre-registered before you look):**
- win = ASR reduction ≥ 0.25 at over-refusal ≤ baseline+0.03, on held-out families.
- each added piece must beat the simpler version on held-out data or it gets cut.

## Step 7 — Stage 3: reasoning trace  *(only after 3–6 hold at prompt encoding)*

- re-run with `enable_thinking=True`. First the free ablation: does `B` move, does invariance change, thinking on vs off?
- then extend the read to `h_{ℓ,t}` across reasoning-token positions; look for harm *recovered* during the trace (intent rises) vs *laundered* (intent stays masked through to compliance).
- **Gate:** reading through the trace must beat the prompt-encoding read on the joint metric, or keep the simpler prompt-encoding version.

---

## The one-line flow

`env → pick Qwen3 + delete old data → build HARD data + split wrapper families → Stage0 band (GATE) → Stage1 divergence (GATE) → Stage2 suppression (GATE) → paired eval (DECISION) → Stage3 reasoning (GATE)`

## The thing to actually do next

Steps 3 and 4 are a day or two of work and they decide everything downstream. You have not run them on any new model yet. Before two axes, suppression heads, trajectories, or reasoning traces — run Stage 0 and look at the band. Every design choice after that is guessing until you've seen the curve.
