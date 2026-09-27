# ACSL — Architecture (rebuilt v3)

*Supersedes prior versions. Changes from v2: (1) the probe reads the **residual stream `h_ℓ` at every layer**, not the attention-sublayer output `o_L` — this is what makes the method architecture-agnostic and portable across token-mixing designs; (2) the build is staged so each step gates the next; (3) reasoning-model base, read at prompt-encoding first and extend through the trace later; (4) an explicit cross-architecture transfer stage (dense attention → DeltaNet/MoE).*

---

## 0. Thesis (unchanged)

Gate the action on an **internal** representation of harm read mid-network, not on surface refusal. Jailbreaks flip the surface while the internal harm encoding largely persists; read the persistent thing and gate on it.

On a **reasoning model** this is *more* true: the reasoning trace is often unfaithful to the actual computation, so reading activations beats reading the trace text. The harm representation evolves *through* the trace, so read position is a design variable (prompt-encoding → during-trace → final-answer).

---

## 1. The tap point — residual stream, every layer (the decision that makes this portable)

Read `h_ℓ` = the **residual-stream activation** at layer `ℓ`, for **all** layers, at the last prompt token (Stages 0–2; extended to trace positions in Stage 3).

Why residual, not the attention output `o_L`:
- `h_ℓ` exists at **every** layer regardless of what the sublayer is (softmax attention, linear/gated attention/DeltaNet, MoE FFN). `o_L` only exists where there's a standard QK attention to hook.
- The difference-of-means direction and steering both live in the residual stream anyway — the attention output is just *added into* it.
- This is the version that can probe a DeltaNet/MoE stack (Qwen3.5) **without redefinition**, which is the prerequisite for the cross-architecture transfer test (Stage 4). Tapping `o_L` would make 3-of-4 Qwen3.5 layers unreadable and kill that experiment before it starts.

Steering also acts on `h_ℓ` (add `g·u` to the residual), so the read and the steer are both architecture-agnostic by construction.

---

## 2. Two axes (the core design commitment)

Two **separate** linear probes over the same `h_ℓ` — not one multi-class head.

- **Intent** `I`: is the framing/request malicious? (benign ↔ malicious)
- **Severity** `S`: how weaponizable is the content if complied with? (inert ↔ dangerous)

| case | intent | severity | gate |
|---|---|---|---|
| benign exploit *explanation* | low | high | answer |
| DAN-wrapped trivial request | high | low | redirect |
| working ransomware request | high | high | refuse/route |
| poem | low | low | answer |

The `low-intent / high-severity → answer` cell is the entire point: legitimate dual-use security work passes while malice is caught. A single graded axis (safe/dangerous/harm) cannot express it, and a graded axis is what let the DAN wrapper drag a score across one boundary in the first run. Each axis is tested for invariance **independently** — two axes only help if *both* reads stay invariant under wrapping.

---

## 3. The staged plan (build in this order; each stage gates the next)

### Stage 0 — establish the invariant band `B`  *(do this first, always)*
On the dense reasoning base (Qwen3), **at prompt encoding**, reading `h_ℓ` at **every** layer:
- per-layer kill test for each axis: `auroc_clean[ℓ]`, `auroc_wrapped[ℓ]` on **held-out** wrapper families.
- `B = { ℓ : delta ≤ τ for BOTH axes AND auroc_clean high }`.
- **Go/no-go:** if no band holds for both axes, two-axis ACSL doesn't work on this model — stop, write the negative result. Expect `B` ≈ 3–5 layers; read the count off the curve, don't pick it.

### Stage 1 — test the interaction  *(cheap descriptive step; gates Stage 2)*
Does intent diverge from severity **across depth under wrapping** — intent suppressed, severity retained — in a way benign prompts don't?
- `div = severity_proj − intent_proj` across `B` (or `Σ(Δs − Δi)`).
- compare the **distribution of `div`** across clean-harmful, wrapped-harmful, benign; compute separability (AUROC of `div` for wrapped-harmful vs benign).
- **Go/no-go for Stage 2:** separable → the suppression signature exists, build the detector. Overlapping → **do not build it**, keep two-axis + trajectory.

### Stage 2 — suppression detector  *(only if Stage 1 passes)*
A **separate head with a different target**: predict *wrapped-vs-genuine*, not harm.
- input: trajectories `[i_B, s_B, Δi_B, Δs_B]` plus co-movement `Δi − Δs`.
- target: `wrapped vs genuine` from `wrapper_family`; **held out by family** (train some families, test on structurally different ones).
- composition: if it fires, **distrust the intent read** and treat intent as HIGH. This catches the suppressed-by-wrapper cases that dominated the first build's gate-misses.

### Stage 3 — reasoning-trace extension  *(only after 0–2 hold at prompt encoding)*
Extend the read from prompt-encoding to the reasoning trajectory: `h_{ℓ,t}` over layers `ℓ ∈ B` and reasoning-token positions `t`.
- harm may be *recovered* during reasoning (intent rises) → detectable; or *laundered* (intent stays masked to compliance) → signature lives in the across-trace movement.
- **Free ablation (Qwen3 thinking toggle):** thinking on vs off, same model — does `B` move, does invariance change?
- **Gate:** the trace read must beat the prompt-encoding read on the joint metric, or keep the simpler version.

### Stage 4 — cross-architecture transfer  *(Qwen3.5: DeltaNet + MoE + multimodal)*
The open question: does the harm representation occupy the **same depth band and geometry** when the token-mixer changes from softmax attention to gated/linear attention (DeltaNet), with MoE routing?
- Qwen3.5 layout is `8 × (3 × DeltaNet → 1 × Gated-Attention)` — 3-of-4 layers have no QK key-match. Because the probe reads `h_ℓ` (residual), it runs on **all** layers unmodified; the `o_L` tap could not.
- **Probe every layer; trace the full progression** — watch whether the harm projection builds through the DeltaNet stretches or jumps at the periodic attention layers.
- **Hypotheses to test, held loosely:**
  - *Same job, same geometry* (the optimistic case): the band and a shared `u` transfer → strong result, "harm representation invariant to token-mixing architecture, not just to jailbreaks."
  - *Same job, different geometry*: signal present but at a different depth / different direction → re-derive `B` and `u` per architecture; still works, but doesn't transfer as-is.
  - *Steering still works regardless*: both read and steer are residual-stream, so steering should function on DeltaNet layers even if the band shifts — verify, don't assume.
- Vision side is genuinely "another head or two": once reading the residual stream, add intent/severity probes that also see image-token positions. The porting cost is the **mixer mismatch**, not vision — and the residual read already neutralizes it.
- **Gate:** report whether band/direction transfer. Either outcome is a result; the transfer case is the stronger paper.

---

## 4. Algorithms

### A — build the axes (offline, frozen base)
```
1. forward each prompt; cache h_ℓ (RESIDUAL stream) at last prompt token for ALL layers.
2. per axis a in {intent, severity}, per layer ℓ:
     u[a,ℓ] = normalize( mean(h_ℓ | a high) − mean(h_ℓ | a low) )      # diff-of-means
     (optionally also train linear probe P[a,ℓ])
3. per-layer kill test on HELD-OUT wrappers -> delta[a,ℓ]               # Stage 0
4. B = { ℓ : delta[intent,ℓ]≤τ AND delta[severity,ℓ]≤τ AND clean high }
5. freeze {u[a,ℓ] : a, ℓ∈B}
```

### B — trajectory read (online, per forward pass)
```
1. forward prompt; capture h_ℓ (residual) for ℓ∈B.
2. x[a,ℓ] = ⟨h_ℓ, u[a,ℓ]⟩
3. i_B=[x[intent,ℓ]], s_B=[x[severity,ℓ]], Δi=diff(i_B), Δs=diff(s_B)
4. div = s_B − i_B                                                      # Stage 1 statistic
5. I*, S* = aggregate(i_B), aggregate(s_B)   # max-over-band first; learned head later
6. suppressed = SuppressionHead([i_B,s_B,Δi,Δs,Δi−Δs])                  # Stage 2, if built
7. return I*, S*, suppressed, div
```

### C — gate
```
if suppressed:  I* = HIGH                       # distrust masked intent read
if I* high and S* high:  refuse_or_route
if I* high and S* low:   redirect
if I* low  and S* high:  answer                  # legit dual-use  <-- the payoff cell
else:                    answer
# thresholds set per over-refusal budget on HARD benign, held out
```

### D — steering (residual, architecture-agnostic; used in deliberation loop)
```
# read -> steer -> re-read, K iters or until |Δr|<eps; fail closed if unresolved
h_ℓ = h_ℓ + g(r)·u[axis,ℓ]      # adds to the RESIDUAL — works on attention AND DeltaNet layers
```

---

## 5. Data schema

```
{ text, intent: 0|1, severity: 0|1|2, wrapper_family: id|null, split: train|heldout }
```
- **Label by intent and content, never by behavior.** A redirect is compliance-shaped behavior on malicious intent → still `intent=1`. Matters more on reasoning models, which redirect more.
- **Sources:** AdvBench harmful (refused) · curated harmful-that-passes (malicious, complied — breaks harm=refused correlation) · working wrappers (split by family) · **HARD** benign: XSTest-style + defensive security text, with **high severity allowed at intent 0** (a benign exploit explanation). The hard high-severity/low-intent rows are what teach the two axes to dissociate; soft benign rebuilds the over-refusal problem.

---

## 6. Novelty (for the writeup)

- **Not novel:** activation probes as a safety screen (Anthropic's Constitutional Classifiers++ deploys a linear-probe screen + ensemble); reading multiple layers (routine).
- **The contribution:** (a) intent × severity as *separately-invariant* residual-stream reads; (b) the **depth-trajectory** (and on reasoning models, the **depth × trace** surface) as the discriminative feature; (c) a **suppression detector** predicting masking from trajectory divergence; (d) the methodological result that the invariant band must be selected **per-axis by kill test, not by clean accuracy**; (e) **cross-architecture transfer** — whether the band/geometry hold across token-mixers (softmax attention vs DeltaNet/MoE). Lead with this conjunction; cite Constitutional Classifiers++ as prior art for the mechanism.

---

## 7. What carries over / what to discard from the old build

- **Keep:** hooks (retargeted to the residual stream), per-layer projection, metrics (paired ASR + over-refusal, bootstrap CIs, AUROC), held-out-wrapper discipline, kill-test logic, frozen-base / tiny-probe pattern.
- **Discard:** AdvBench-safe-only negatives (replace with hard negatives), the old single-axis head, all Qwen2.5-instruct numbers, and any tap built specifically on the attention output `o_L` (move to residual `h_ℓ`).
- Runs frozen-base + tiny-probe on the 16 GB card. Qwen3-4B dense, text-only, fp16 fits with room for activation buffers. Qwen3.5 is a later *port* (Stage 4), enabled by the residual read.

---

## 8. The discipline that makes any of this a result

Pre-register prediction + decision rule + kill criterion before each stage. Report ASR and over-refusal **together** (refuse-all trivially scores ASR 0). Hold out wrapper families end-to-end. Ablate every added piece — two-axis must beat one-axis, trajectory must beat aggregate, suppression head must beat two-axis+trajectory, trace read must beat prompt-encoding read, and the cross-architecture claim must be shown, not assumed. Drop anything that doesn't clear its bar.

---

## 9. Build order (one line)

`Qwen3 dense, residual read, all layers → Stage0 band (GATE) → Stage1 divergence (GATE) → Stage2 suppression (GATE) → paired eval (DECISION) → Stage3 reasoning trace (GATE) → Stage4 port to Qwen3.5, test band/geometry transfer (RESULT)`

The unmoved prerequisite: run Stage 0 on Qwen3 and look at the band before building anything downstream. Every stage after it is a hypothesis until that curve exists.
