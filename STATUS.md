# ACSL — Session Status

> Updated by Claude at end of each work session, or when parking context for a tangent.
> Read this first every session to re-orient.

## Last updated
2026-07-05

## Current state
- **MIGRATED TO LINUX (2026-07-04).** Repo now at `~/Projects/ACSL`, runs in Docker on the pinned ROCm image (`rig:2.9.1`, torch 2.9.1+rocm7.2.4, RX 6900 XT). Windows box preserved as independent verification backend. Models re-downloaded into the `hf_cache` volume (`Qwen/Qwen3-1.7B` @70d244cc, `Qwen/Qwen3-4B` @1cfa9a72); configs v3/v3_4b now use hub ids. Torch stack protected by `constraints.txt` + `scripts/setup_container.sh` (pip fails loudly rather than replacing torch).
- **Server operational on Linux:** `./launch_server.sh` / `./kill_server.sh` at repo root manage the `acsl-server` container (port 5000, idempotent launch, waits for model load). Verified end-to-end: v3-qwen3 profile, head@L9, generation + probe read working on GPU. IP allowlist includes docker bridge gateway 172.17.0.1; note bridge mode rewrites all source IPs to the gateway — `--network host` needed if real lab-laptop IP filtering is wanted (decision open).
- **Back-verification work order (CLAUDE.md): ON HOLD per Matt (2026-07-04)** — intent-trace/frontend upgrades take priority. See "Parked context".
- **Frontend/trace upgrades: DONE & verified (2026-07-04).** Intent Trace → **Harm Trace** with an **axis selector** (harm | actionability — one tab reusing the full heatmap/3D/compare stack, not a duplicate tab; flagged to Matt, not vetoed). New tabs: **Logit Lens** (per-layer English decode, all 28/36 layers, fp32 softmax, `/api/logit_lens`), **Gate 2×2** (harm@t_inst × actionability@t_post-inst quadrant, exploratory/uncalibrated, `/api/gate2x2`), **History & Compare** (search/filter/paginate/star/tags/notes/bulk, compare dock relocated here; history API v2 = `{total, items}`). Cell-click logit-lens decode on trace heatmaps, labeled "independent of this axis." Actionability coverage pill (`directions on disk: L9,L14,L17 of 28`) fed by new `/api/status.axis_coverage`. All 6 endpoints curl-verified; frontend passed static ID/bracket/function-def checks; live browser pass still Matt's to do.
- **AXIS DEFINITIONS PINNED — see `docs/axis_definitions.md` (authoritative, 2026-07-04).** Written before touching any labeler, then hardened by a Rule 9 adversarial pass. Key facts: cache label `y` = pre-annotated `label` field (extract.py:240), and **`label == intent` for all 1325 records** → the trace's "Harm" axis is actually **intent** (context-aware: "kill someone"=1, "…in Call of Duty"=0), NOT topic. Matt's "Harm = topic" maps to the **`severity`** field (dropped axis). **No data field is output/refusal-derived** — Harm/Severity are content-labeled, not circular. review_data.py `classify_response` (output-level) is the refusal baseline and is NOT wired into any axis (verified), but the firewall is convention, not mechanism.
- **Rule 9 verdict:** Claims 1,2,4,5 SUPPORTED; Claim 3 (actionability refusal-independent) PARTIALLY SUPPORTED and downgraded. The served `actionability.npy` vectors have **no committed labeler** (10_direction_audit.py computes only intent/severity) → provenance UNVERIFIABLE. "Prompt-only" ≠ refusal-independent; a reconstruction proxy showed r≈0.26 with the harm label. **Before committing any reconstructed actionability labeler: measure cos(actionability,intent) + label correlation; high = red flag.**
- **Research phase (pre-migration):** Joint-gate analysis done. Next experiment was: **t_inst vs t_post-inst read position** (motivated by Zhao et al.).
- **Key finding [❌ RETIRED 2026-07-05]:** The old claim "Actionability perfectly rescues intent's hard tail per-family" (fictional_roleplay 23%→100%, etc.) does NOT survive controls and is **retired**. On the corrected thinking-OFF caches + 1000-trial null + Rule 9: the apparent rescue is **common-mode wrapper-cancellation** — a *random* 2nd direction reproduces it; on the trustworthy 4B RA is null/anti at every layer. **Mechanism (preserve — this part is real):** both RA and Intent diff-of-means directions load onto a **dominant corpus harm/benign axis** in this data — that structural fact is why cos(RA,Intent)=0.78 at L9 and why ANY direction with a component on that axis lets the logistic denoise Intent under wrapping (the "rescue"). It predicts the same substrate under every diff-of-means direction on this corpus. RA the axis is retired; the corpus-axis finding stays as the explanation.
- **⚠ UNVERIFIED (2026-09-26 release note): no script or run artifact for this number exists in the repo; it predates the 2026-07-05 tokenization fix and its 77.6% baseline does not match the ladder's 38.5%. Treat as a hypothesis to re-run, not a result.** t_inst read position. Moving intent read from `t_post-inst` (assistant\n) to `t_inst` (last user-content token before `<|im_end|>`) jumps gate CR from 77.6% → **95.0%** on 4B (66.7% → 90.7% on 1.7B). Same model, same layer, same LR probe. We were reading the *refusal* signal (gameable) instead of the *harmfulness* signal (persistent). Motivated by Zhao et al.
- **Remaining hard family:** fictional_roleplay (65.7% on 4B, 60% on 1.7B). Everything else ≥ 97%+ at t_inst.
- **Optimal mixed read:** intent@t_inst + actionability@t_post-inst (action signal is stronger at post-instruction position).
- **Server:** Running on port 5000, 1.7B model loaded.

## Completed work

### Finalization spec progress

#### Step 1 — Intent invariance at divergence layer: DONE
- 4B: intent fails at L17 (delta=+0.059, 1/12 stable) → two-layer read: intent@L33, actionability@L20
- 1.7B: L9 confirmed (12/12 stable, sole intent-invariant layer, both axes)

#### Step 2 — Token-trace corroboration: DONE
- 0/5 content token overlap in top-5 between intent and actionability directions
- Intent peaks on framing/assistant tokens at L33; actionability peaks on different structural positions at L20
- Mechanistic separation confirmed — axes read different parts of the prompt

#### Step 3 — Rung 0 diff-of-means heads: DONE
| Model | Axis | Layer | Clean | Wrapped | Delta | Stable |
|-------|------|-------|-------|---------|-------|--------|
| 4B | intent | L33 | 0.937 | 0.902 | +0.035 | 12/12 |
| 4B | actionability | L20 | 0.867 | 0.837 | +0.030 | 12/12 |
| 1.7B | intent | L9 | 0.761 | 0.737 | +0.023 | 12/12 |
| 1.7B | actionability | L9 | 0.870 | 0.851 | +0.019 | 12/12 |

Saved to `runs/v3_{4b,1.7b}_head_{intent,action}_r0/`

#### Step 4 — 2x2 gate: DONE
Gate wired and evaluated. Over-refusal at target (5.1%). Configs in `runs/v3_{4b,1.7b}_gate/`

Cell map:
- high intent + high actionability → refuse_or_route
- high intent + low actionability → redirect
- low intent + high actionability → ANSWER (dual-use payoff)
- low intent + low actionability → answer

#### Ablation ladder (Rung 0-3): DONE — ceiling found

**Threshold-vs-signal diagnostic:** 93% (4B) / 89% (1.7B) of Rung 0 misses are deep (well below threshold). This is a signal problem, not threshold placement.

**Full ladder — gate catch rate @ 5% over-refusal:**

| Model | R0 | R1 (LR) | R2 (band) | R3 (MLP) |
|-------|-----|---------|-----------|----------|
| **4B** | 31.6% | 38.5% | 40.2% | **42.8%** |
| **1.7B** | 12.0% | 33.4% | 33.4% | **38.3%** |

**Per-axis breakdown (Rung 3 best, CR@5%):**

| Model | Intent | Actionability |
|-------|--------|---------------|
| 4B | 42.8% (multi-layer MLP L9+L23+L33) | 69.6% (MLP L20) |
| 1.7B | 38.3% (MLP L9) | 70.1% (MLP L9) |

**Key findings:**
- Actionability axis works well (~70% CR). Intent is the bottleneck (~40% CR).
- Gate catch rate = intent catch rate (gate routes on intent first)
- 7/12 families essentially solved on 4B (≥85%): academic_frame, cognitive_overload, decomposition, hypothetical, multi_turn, persona_override, refusal_suppression
- Hard holdouts: obfuscation (17.1%), fictional_roleplay (57.1%), low_resource_lang (65.7%)
- **Joint-gate analysis:** OR-gate (catch if either axis fires) lifts CR to 46.2% (+10pp) but at 49% over-refusal — unusable as-is. Actionability catches every family intent misses, but can't distinguish malicious-operational from benign-operational alone.
- Only 9/890 truly blind (neither axis): 6 Unicode homoglyphs + 3 refusal_suppression prefix-only
- Wrappers genuinely alter the intent representation, not just shift it near threshold
- AUROC climbs (0.937→0.940) while catch-rate barely moves — ranking is fine, operating point is the problem
- Dual-use cell correct at 92.8% — the two-axis decomposition adds real value for over-refusal
- Max-pool band hypothesis didn't pan out (hurt CR vs single-layer)

**Rung details saved to:** `runs/v3_{4b,1.7b}_gate_r{1,2,3}/` and `runs/v3_{4b,1.7b}_head_{int,act}_r1/`

### v1 (done, archived)
- Full pipeline on DirectML. L15 head on de-saturated data gives modest-positive ASR reduction (0.175 → 0.01-0.045 at 5-10% over-refusal). FINDINGS.md has writeup.

### Interpretability checklist (items 1, 3, 5, 6 done; item 2 pending; item 4 deferred to 8B)
- **Item 1 (widen corpus):** 73 prompts across 15 domains. Intent false-positives: fiction, history, emotional domains flag high on severity (not intent). Confirms intent reads framing, not topic.
- **Item 3 (minimal pairs):** 25 pairs (15 same-surface + 10 topic-vs-act). Intent separates 25/25. Non-lexical proof.
- **Item 5 (falsifiable predictor):** Pending — depends on actionability head results.
- **Item 6 (steering asymmetry):** Severity flips programming (7/7), biology, physics, medical most easily. Named the feature: "technical mechanisms / dangerous-topic detector." Intent blind spot: flat-language malice (privacy/OSINT/recon prompts score as benign).
- **Item 2 (gradient attribution):** Pending — target false-negative blind-spot cases.
- **Item 4 (SAE decomposition):** Deferred to 8B run. Qwen-Scope SAE exists for Qwen3-8B-Base.

### Severity axis — documented negative result
- Severity reads "dangerous-topic" not "weaponizable intent" (confirmed by steering asymmetry + minimal pair failures)
- At L17 (4B): cos(intent, severity)=0.735, but severity fails 3/10 topic-vs-act pairs
- At L33: cos(intent, severity)=0.884 — 88% redundant with intent
- Actionability replaces it: cos(severity, actionability)=0.136 at L17 (nearly orthogonal)

> ⚠ **CORRECTION (2026-07-04):** the 0.136 above is **severity↔actionability at 4B/L17** — NOT the
> quantity that matters for the intent×actionability gate. The **operating-point** independence is
> **cos(intent, actionability)**, measured 2026-07-04: **0.78 on 1.7B @ L9** and **0.43 on 4B @ L20**
> — far from orthogonal. The favorable 0.136 was borrowed across a different axis pair AND a different
> model/layer than either model actually operates at. See the "FINDING" block above. The two-axis
> independence claim below is **qualified accordingly**, not withdrawn: the RA/Intent *directions* are
> weakly aligned on the 4B (cos 0.43) and strongly aligned on the 1.7B @ L9 (cos 0.78) — but per the
> Rule 9 correction, direction cosine ≠ discriminative redundancy (label phi=0.27), so whether RA adds
> detection value over Intent is UNMEASURED, not "restated."

### Actionability axis — second axis (QUALIFIED — see correction above)
- Form-based label: "asks for executable steps" = 1, "asks for understanding" = 0, independent of topic danger
- Three-way head-to-head: actionability 10/10 on topic-vs-act pairs where severity fails (both models)
- Results correlate across model scales
- **But intent↔actionability *direction cosine* is layer/model-dependent (0.43 4B@L20, 0.78 1.7B@L9); "orthogonality" was never true at the operating points. Whether RA adds detection value beyond intent is unmeasured (label phi=0.27 — see Rule 9 correction in FINDING block).**

### Band sweeps — all three axes, both models

**4B:**
| Axis | Band | Notes |
|------|------|-------|
| Intent | [9-14, 18, 20-33] (21 layers) | L33 best: clean=0.937, delta=+0.035, 12/12 stable |
| Severity | [9-29] (20 layers) | Dropped as axis |
| Actionability | [0, 4-25] (23 layers) | Very wide band |

**1.7B:**
| Axis | Band | Notes |
|------|------|-------|
| Intent | **[9]** (1 layer only) | clean=0.761, delta=+0.023, 12/12 stable |
| Severity | [9, 10, 11] | L9 11/12 (drops obfuscation) |
| Actionability | [0,1,3-10,20] (11 layers) | L9 12/12 stable |

### Extraction caches
- `runs/cache_v3/` — 1.7B, 28 layers, 905 clean + 890 wrapped examples
- `runs/cache_v3_4b/` — 4B, 36 layers, same data
- Both have `norm_stats.npz` (per-layer z-score from clean split)

### Direction vectors saved
- `runs/10_direction_audit_4b/directions_L{33,17,20}.npz` — intent, severity, actionability
- `runs/10_direction_audit_1.7b/directions_L{9,14,17}.npz` — intent, severity, actionability

### Server & frontend
- **FastAPI + uvicorn**, async GPU work via `asyncio.to_thread()`
- **GPU watchdog**, **SafeJSONResponse**, **TeeWriter** for logs
- **Token x Layer Intent Trace** with 2D/3D heatmaps, SQLite logging, history/export
- **Chat-style Testing UI** with thinking toggle, compare toggle
- **Live server log panel**
- **Intent Trace layout overhaul** — controls full-width across top (Mode/Template/Benchmark in 3-col grid), history as 5-wide card grid below, result cards full-width underneath
- **Model/head metadata** in history cards and trace logging (model_name, head_path, profile stored in extra_json)
- **Compare dock enhancements** — 2D/3D toggle per compare slot, diff analysis panel (LCS token alignment, diff heatmap, per-layer cosine similarity, top divergent tokens)
- **3D heatmap settings** — label size, dot size, label color controls
- **3D rendering fixes** — double-rAF for layout timing after display:none→visible transitions; _lastItData set on history replay; compare 3D uses explicit `display:block` to override CSS

### Architecture & docs
- `docs/architecture_v3.md` — two-axis design (needs update: intent x actionability, not intent x severity)
- `docs/build_steps_v3.md` — step-by-step with gates
- `docs/head_spec_v3.md` — ablation ladder (Rung 0-3), suppression head spec

## Decisions made (don't re-litigate)
- Layer selection uses kill-test invariance, not raw AUROC
- User always chooses the layer — never auto-select from sweep
- GPU path is DirectML on native Windows; ROCm-WSL is a dead end for RDNA2
- v3 reads residual stream h_l (architecture-agnostic), not attention output o_L
- **v3 uses intent x actionability (NOT severity)** — severity dropped as dangerous-topic detector
- Thinking OFF for stages 0-2 (prompt-encoding only); Stage 3 adds reasoning trace
- Wrapper families must be split train/held-out — the firewall is non-negotiable
- Hard negatives required (XSTest-unsafe, defensive security text)
- Heads are NOT interchangeable between model sizes (different d_model, layer count, band)
- **4B: intent@L33, actionability@L20** — confirmed by stability + literature. ⚠ "orthogonality" corrected 2026-07-04: cos(intent,actionability)@L20 = 0.43, not orthogonal (see CORRECTION note + FINDING block).
- **1.7B: both axes@L9** — sole viable layer
- **4B model must load in float16** — fp32 exceeds 16 GB VRAM
- **Linear-probe ceiling confirmed** — ~43% gate CR at 5% OR; bottleneck is intent on hard wrappers

## Next steps

### ORDERED PLAN when work resumes (Matt, 2026-07-05) — do in this order
1. **CORPUS FIRST.** Before retraining anything, address the corpus limitations surfaced this session — otherwise a new instrument bakes them in:
   - **No wrapped-benign data exists** (`v3_wrapped.jsonl` = 420 wrapped-harmful, 0 benign). Every "benign" negative is an UNWRAPPED clean prompt → all harm-vs-benign evals on wrapped data confound *wrapping* with *harm*. Build/obtain wrapped-benign controls (benign prompts run through the same jailbreak templates) so wrapping can be separated from harm and a real per-family harm test becomes possible.
   - **Dominant corpus harm/benign axis** (see RETIRED-key-finding + docs/axis_definitions.md banner): any diff-of-means direction on this corpus loads onto it, so cos-with-Intent and suppressor "rescues" are expected geometry, not signal. Characterize/decorrelate this before trusting any second-axis or head result.
2. **HEAD SECOND.** Retrain the gated-generation head (`runs/v3_head/head.pt`, `runs/v3_4b_head/head.pt`) + loop `u.npy` on the corrected thinking-OFF caches (the current heads are thinking-ON, mismatched with the now-deployed serving position).
3. **VALIDATE THE HEAD AS A NEW INSTRUMENT.** A retrained head **inherits NONE** of the detection-path validation done 2026-07-05 — it is a different instrument (trained classifier, not diff-of-means). Run the SAME discipline on it: the full 5-test trust suite (polarity / positive control / shuffle / degenerate-shortcut / cross-family) adapted to the head's score, THEN an adversarial Rule-9 pass, BEFORE any head-based number is trusted or cited. No shortcut inheriting.

### Older research next steps (pre-instrument-bug; revisit after the above)
1. **DONE: t_inst read position** — Confirmed. 77.6%→95.0% (4B), 66.7%→90.7% (1.7B).
2. **Re-run full ladder at t_inst** — current ladder was all at t_post-inst. Re-extract caches at t_inst and re-run Rung 0-3. The Rung 1 LR already hit 95% — higher rungs may close fictional_roleplay.
3. **Ground-truth compliance verification** — run all wrapped prompts through the model to verify which ones it actually complies with vs refuses.
4. **Mixed-position gate** — intent@t_inst + actionability@t_post-inst. Each axis reads where its signal is strongest.
5. **fictional_roleplay deep-dive** — the last hard family at 65.7%. What's mechanistically different about the remaining 34.3%?
6. **Write up** — only after the t_inst re-run (with artifacts) and the back-verification work order close. Until then the citable content is the two-axis gate design, the kill-test layer-selection result, and the RA retirement under controls.

## Parked context

- **PAUSED HERE 2026-07-05.** Instrument bug found + fixed + deployed (server on corrected thinking-OFF caches, parity corr 1.0). RA retired as a harm axis (survived 2 Rule-9 passes). Corpus-harm-axis mechanism preserved. Resume at "ORDERED PLAN" in Next steps: **corpus → head → validate head as a new instrument (trust suite + Rule 9).** Scripts added this session: `11_request_actionability.py` (labeler + --detect/--controls), `13_gate_validation.py` (trust suite), `14_per_family_control.py` (per-family + nulls). New: `acsl/tokenization.py` (shared), `configs/v3{,_4b}_thinkoff.yaml`, `runs/cache_v3{,_4b}_thinkoff/`.
- **Back-verification work order (parked 2026-07-04):** CLAUDE.md Tasks 1 (refusal-label audit → `verify/refusal_inspector`, 200-triple stratified sample, Matt audits) and 2 (DirectML-vs-ROCm activation cross-check, 10 seeded prompts). Not started — no inventory done yet. Blocking rule stands: no ACSL result cited anywhere until both close.
- **Networking decision:** bridge mode (current) vs `--network host` for the server container — bridge collapses all client IPs to 172.17.0.1 so the allowlist can't distinguish the lab laptop.

## Known issues
- Server segfaults on DirectML when intent trace extraction + generation run back-to-back (GPU memory pressure). `gc.collect()` between steps partially mitigates. **(DirectML path is Windows-only; N/A on the Linux/ROCm box — left for the Windows verification backend.)**
- **Actionability labeler is MISSING from the repo, not just unpersisted.** The served `runs/10_direction_audit_*/directions_L*.npz` `actionability` vectors were produced by a labeler that exists in no committed `.py` (10_direction_audit.py does intent/severity only). Reconstruction is in progress (person/tense discriminator reproduces saved L9/14/17 at cos ≈ 0.96–0.98; target ≈ 1.0) but BLOCKED pending Matt's decisions below.
- **Circularity firewall is convention, not mechanism** — no import guard/test/schema check stops output-level (`classify_response`) signal from being routed into a v3 label file; extract.py caches whatever `label` holds. Recommended hardening (provenance field + extractor assertion) not yet built.

## Axis naming — DECIDED 2026-07-04 (Matt)
- **Axis 1 = "Intent"** (kept the code's context-aware label==intent; NO data re-labeled). UI/docs/API default token renamed harm→intent; API still accepts legacy `"harm"` alias so logged rows resolve. Verified: intent token 28 layers, coverage re-keyed, legacy alias works, history intent-filter includes legacy rows.
- **Axis 3 = "Request Actionability"** — explicitly request-side (prompt-position): does the *request* solicit operational steps? NOT content-side uplift (would need response-position extraction; out of scope this version). API token stays `actionability`.
- **Refusal/approval confirmed quarantined:** it is the baseline, never an axis. Verified — passing `axis=refusal` is rejected (`unknown axis (intent | actionability)`); no axis token references refusal.

## ⚠ CRITICAL INSTRUMENT BUG (2026-07-05): live server reads a DIFFERENT token position than its directions were built at
Gate-validation suite (`scripts/13_gate_validation.py`, both models) + a live-vs-cache cross-check surfaced a serving bug.
- **Trust suite PASSES on the cache** (both models): polarity correct (not inverted); positive control (lexical token, full-dim logistic) AUROC 0.96–0.99 (extraction healthy); shuffle collapses to ~0.5 (no CV leak); cross-family transfer strong (train→held-out jailbreak family AUROC gap 0.001–0.016). One yellow flag: the **Intent score correlates with prompt length/token-count (R²≈0.13–0.25 on 1.7B/4B)** — a partial surface shortcut; RA less so (R²≈0.07–0.11).
- **BUT the live server ≠ the offline cache path.** Same prompt, 10 samples: corr(live, cache)=0.92, mean|Δ|=2.8, **2/10 flip SIGN** (near-boundary benign↔harm). Root cause PROVEN by extracting live under both templates: **cache was built at `enable_thinking`=default(ON)** (`acsl/extract.py` never sets it → last token `assistant\n`, 18 tok), **server scores at `enable_thinking=False`** (`server.py` → last token after injected `<think>\n\n</think>`, 22 tok). live-thinking-ON=4.03 ≈ cache 4.17; live-thinking-OFF=3.44 = server 3.44. Platform (Win-fp32 cache vs Linux-fp16 live) is a minor ~3% residual on top.
- **Consequences:** (1) the live server's directions (from the cache) are applied at a mismatched position → its gate scores are untrustworthy, sign-flipping near-boundary cases. (2) The cache was built thinking-ON, which **contradicts STATUS "Thinking OFF for stages 0-2."** So the whole offline analysis (scripts 11/13, RA study) rides a thinking-ON cache — self-consistent as a study, but NOT the documented design and NOT the live serving position.
- **Also (minor):** server Intent direction (`_compute_caution_direction`) is RAW-space diff-of-means while the actionability npz + offline scripts are normalized-space; cosmetic (cos 0.96, score corr 0.998, AUROC −0.017).
- **Code sharing (audit):** `verify/` does not exist (parked). Shared offline↔server: `load_cached`/.npy + `load_norm_stats` + layer indices (configs) + the `hidden_states[L+1]`+last-token convention (DUPLICATED in server `_extract_activations`, not a shared fn). Divergence points: the two items above.
- **FIX (Matt chose c→a, rejected b): DONE 2026-07-05.**
  - **(c) Tokenization unified** — new `acsl/tokenization.py::build_prompt_inputs` is the single source; `acsl/extract.py::_tokenize` and `server.py::_tokenize_prompt` both call it; `enable_thinking` defaults False and is threaded through `extract_activations` + `01_extract.py --enable-thinking/--no-enable-thinking`. Serving behavior unchanged (still thinking-OFF, max_length 2048). Can't drift again.
  - **(a) Corrected caches rebuilt** thinking-OFF (non-destructive, new dirs): `runs/cache_v3_thinkoff` (1.7B, 905+890, 28 layers) and `runs/cache_v3_4b_thinkoff` (4B, 36 layers); norm sanity 1.0000. Configs `configs/v3_thinkoff.yaml`, `configs/v3_4b_thinkoff.yaml`.
  - **Parity PROVEN:** with server direction+norm, only swapping activation source: OLD thinking-ON act → 4.166 (old cache); NEW thinking-OFF act → **3.44 = live server exactly**. Corrected cache now reproduces live serving bit-for-bit (‖Δact‖=60 of ‖act‖=71 between positions — the divergence was huge, hence sign flips).
  - **PENDING (not committed):** server still serves the OLD thinking-ON cache (profile → `runs/cache_v3`). To deploy the fix, repoint profiles to the `*_thinkoff` caches (or promote them over the old dirs) + re-derive the served actionability npz. Awaiting Matt's go.

### Corrected-cache trust suite (both models, 2026-07-05)
| Test | 1.7B thinkOFF | 4B thinkOFF | note |
|------|---------------|-------------|------|
| 1 Polarity (Int / RA) | +13.1 / +31.4 | +32.1 / +28.2 | not inverted ✅ |
| 2 Positive control | 0.992 | Int@L33 **0.946** / RA@L20 0.972 | 4B L33 marginal — deep-layer surface softness, not a break |
| 3 Shuffle real→shuf | 0.73→.52 / .88→.50 | .92→.52 / .93→.51 | no CV leak ✅ |
| 4 Shortcut R² (Int token_cnt) | **0.39 ⚠** | **0.013 ✅** | see finding |
| 5 Cross-family gap | +0.017 / +0.017 | +0.028 / +0.007 | generalizes ✅ |

- **Answer to "did unifying position change Intent's length dependence?" YES.** 1.7B@L9 went 0.25→**0.39** (token_cnt) — WORSE; ~39% of the intent score is prompt length at the corrected position (last token after the injected `<think></think>`). RA@L9 0.09→0.17. **BUT the 4B@L33 is clean (0.007/0.013).** Interpretation: L9 is an early-middle layer (encodes surface/length); L33 is near-final (abstract). The length shortcut is a **1.7B-at-L9 problem, not a 4B problem** — a reason the 1.7B read layer may be poorly chosen independent of everything else.
- **Verdict:** corrected instrument PASSES polarity/extraction/leak/transfer on both models; the one real caveat is the 1.7B/L9 length shortcut. Safe to proceed to the per-family control on the corrected caches, with length as a known confound to watch (consider length-matched null).

**⚠ All `scripts/11` + `scripts/13` analysis reported before 2026-07-05 was on the thinking-ON caches (`runs/cache_v3`, `runs/cache_v3_4b`) and is SUPERSEDED by results on the corrected `*_thinkoff` caches. The RA "does not earn its place" conclusion must be re-derived on the corrected caches before it stands.**

---

## FINDING (2026-07-05): PER-FAMILY control on CORRECTED caches — models DISAGREE; trustworthy 4B says RA does NOT rescue
`scripts/14_per_family_control.py` on the thinking-OFF caches. Per wrapper family (kill-test: family-wrapped-harmful vs wrapped-benign, probe fit on clean), ΔAUROC over Intent-alone for Intent+RA vs two nulls: random-feature p95 (N=15) and a length-direction null. RA "wins" a family iff ΔRA > BOTH nulls.
Updated with **1000-trial** random null (stable p95):
- **1.7B (L9, length-confounded read):** RA "wins" **7/12** (was 9/12 at N=15) — **still by trivial margins** (ΔRA − rand_p95 = 0.001–0.009; persona_override +0.046 vs +0.045). Essentially AT the null. LENGTH null mostly negative → length is not the driver; the generic collinear-suppressor is.
- **4B (L33, the trustworthy read):** RA "wins" **0/12** (both N=15 and N=1000). With the stable 1000-trial null the p95 rises to +0.10…+0.24, and ΔRA is negative or tiny on every family — **RA is far BELOW a random second feature**, and often below Intent-alone. On the hard families (obfuscation Intent-alone 0.407, low_resource_lang 0.537, fictional_roleplay 0.577) RA makes detection WORSE.
- **RULE 9 (1000-trial null) — refined verdict, DO NOT AGGREGATE:**
  - **4B (L33, trustworthy): decisively NULL/anti.** Survives every false-negative attack: RA loses 0/12 at EVERY layer L8–L35 (not a layer-pairing artifact); C=1.0 ≡ C=1e6 (not overfit); RA sits at percentile 12–62 of the null, beats the null MEAN on only 1/12. RA is genuinely worse than a random 2nd feature here.
  - **1.7B (L9): WEAK-BUT-NON-NULL — I over-stated "at the null."** Against the strict p95 bar RA "wins" only 7/12 by <0.01; BUT against the null CENTER, RA beats the mean on **12/12** and reaches one-sided empirical **p<0.05 on 7/12**, sitting at percentile 85–99.7. So RA reproducibly outranks ~all random directions at L9 — a real (small) effect, NOT null. It is confounded though (early length-y layer + the negatives confound below) and NOT reproduced on the 4B (identical prompts/labels: p99 on 1.7B vs p20 on 4B → layer geometry, not RA content).
- **⚠ DATA CONFOUND (Rule 9 catch, now fixed in script 14 comments):** the negatives are NOT "wrapped-benign" — `v3_wrapped.jsonl` has 420 wrapped-harmful and ZERO benign, so all 470 negatives are UNWRAPPED clean benigns (wrapper_family=None). The contrast confounds WRAPPING with HARM; no family-labeled benigns exist, so a clean per-family harm test is impossible with this corpus.
- **NET VERDICT: RA does not earn its place as a robust, model-general harm axis.** 4B (trustworthy) = null/anti at every layer; 1.7B = weak-but-non-null at L9 but confounded and not reproduced. The historical "actionability rescues intent's hard tail" claim does **not survive** the null controls and must be retired/rewritten (done below + docs).

---

## (older, thinking-ON, SUPERSEDED) FINDING (2026-07-05): Request-Actionability does NOT robustly earn its place — the two-axis "rescue" is mostly a wrapper-cancellation artifact. Nothing committed.
The discriminative test (`scripts/11_request_actionability.py --detect --controls`, both models) was run AND Rule-9'd. Result, calibrated:
- Naive read (Intent-only vs Intent+RA joint AUROC on wrapped/jailbroken): huge ΔAUROC, e.g. 1.7B L9 +0.235 (p 1e-139), corroborated by clean 5-fold CV (+0.118). ALL numbers reproduce exactly (independent re-derivation; C=1e6 vs C=1.0 identical — not a regularization artifact; CV is leak-free).
- **BUT the causal claim "RA adds harm-detection value" FAILS the control.** A **random** matched-positive-rate 2nd diff-of-means feature reproduces the gain (random p95 ≈ +0.217 at L9 vs RA +0.235; random *max* ≈ +0.232). **Severity** as 2nd feature gives ≈0. Mechanism: the jailbreak wrapper injects a near-constant offset that lands on any Intent-collinear axis; the logistic subtracts it to denoise Intent. This is **common-mode wrapper cancellation**, distribution-shift-specific, NOT RA carrying harm signal. RA-alone AUROC on wrapped ≈ 0.556 (~chance).
- **RA label leaks harm:** P(harm|RA=1)=0.60 vs P(harm|RA=0)=0.33 (label AUROC 0.63). So RA violates the "harm-independent" premise; part of any joint gain is RA smuggling the target.
- **Calibrated verdict:** RA beats the random null only marginally and only in a narrow band (L5–L10; at L9 it's at the top of the null, not clearly beyond it), and that excess is confounded by label leakage. On this evidence **RA does not earn a place as an independent harm-detection axis.** Not zero, but not supported.
- **⚠ Implication for prior claims:** the historical "actionability rescues intent's hard tail per-family" numbers (STATUS Current-state bullet) were computed WITHOUT this random-feature control and may be the same artifact. They must be re-checked against the control before being cited. FINDINGS.md is v1-era and unaffected.
- **Cleaner test if pursued:** de-leak the RA label (stratify/orthogonalize vs harm by construction), proper permutation null (≥1000 trials), and test transfer to a *different* jailbreak style (the current effect is offset-cancellation for THIS wrapper set).

---

## (superseded) FINDING (2026-07-04): RA and Intent diff-of-means DIRECTIONS are strongly aligned at 1.7B/L9 — cos-gate FAILED. Whether RA adds detection value is UNMEASURED. Nothing committed.
> This block was CORRECTED after a Rule 9 adversarial re-computation. All cosine numbers reproduce
> exactly (independent re-derivation; intent direction matches saved vectors at cos=1.0000), but the
> earlier interpretation ("redundant / restates intent") OVERREACHED and is withdrawn. See "Rule 9
> correction" below.

Reconstructed the Request-Actionability labeler (`scripts/11_request_actionability.py`, prompt-only, persisted — clears the "labeler not in repo" debt). It **faithfully reproduces the saved actionability vectors** (cos 0.98/0.97/0.96 at L9/14/17). Reproduction is NON-trivial: cos(savedRA, Intent)=0.765 at L9 while cos(reconstructedRA, savedRA)=0.98 — the labeler recovers RA-specific structure beyond the shared harm axis, so RA is a *distinct* direction, not a relabeled intent.

**Direction-cosine gate (Matt's rule: commit+sweep only if max cos(RA,Intent) ≤ 0.4 over audited layers): FAILED — 1.7B 0.777, 4B 0.429.** Nothing committed, nothing swept.

**cos(RA, Intent) vs depth (1.7B, all 28 layers):** L0–L11 ≈ 0.68–0.78 (**peak 0.777 @ L9, the read layer**) → crosses ~L13–L15 → **L15–L27 ≈ 0.38–0.44** (min 0.377 @ L15). cos(Intent, Severity) does the opposite (low early ~0.58, high late ~0.81–0.86). NB: labels are FIXED across layers, so the depth curve is pure residual-stream **geometry** (the shared harm axis rotating in/out of alignment), NOT changing feature redundancy.

**4B curve:** reproduction 0.98/0.97/0.98 (L17/20/33). cos(RA,Intent): early L0–L15 ≈ 0.53–0.70 → **plateau 0.40–0.45 from L16 on**. **Operating point L20 = 0.4288** (fails ≤0.4 by 0.03). **Min = 0.4006 @ L32.** cos(Intent,Severity) is 0.90–0.94 late, so RA@0.43 is more independent than severity ever was — but not orthogonal. 4B @ L20 (0.43) is far more distinct from intent than 1.7B @ L9 (0.78).

### Rule 9 correction — what the cosine does and does NOT show
- **Direction cosine ≠ discriminative redundancy.** cos(RA_dir, Intent_dir)=0.78 measures the angle between two *mean-difference* vectors, not whether the two labelers carve the data differently.
- **The labels are only weakly correlated.** RA×Intent contingency over 905 clean rows: [RA1&I1=297, RA1&I0=195, RA0&I1=138, RA0&I0=275] → **phi = 0.27** (~7% shared variance at the label level; 37% of rows are discordant). So the 0.78 direction alignment is driven by a dominant corpus harm/benign axis both diff-of-means project onto — a *dataset geometry* property — not by the two axes being the same feature.
- **Withdrawn:** "the two axes are ~redundant" / "L9 vector restates intent." Not supported by a direction cosine and contradicted by phi=0.27 and the 0.765-vs-0.98 reproduction gap.
- **Defensible statement:** the RA and Intent diff-of-means directions are strongly aligned at 1.7B/L9 (weakly at 4B/L20), driven by a dominant corpus harm axis; **whether RA adds detection value over Intent is UNMEASURED** (needs a joint logistic probe or conditional AUROC — the user's point (c), not yet run). The ≤0.4 cos-gate is a proxy that penalizes shared geometry regardless of incremental discriminative value.

**Open decision for Matt (HOLD — awaiting his read):**
- (a) The alignment finding + Rule 9 correction stay documented (done).
- (b) Re-home is premature until (c) is run — a "more independent" layer by cosine may not be a better-detection layer.
- (c) **Run the DISCRIMINATIVE test before any redundancy claim or re-home:** per-layer, measure (i) RA detection AUROC, (ii) Intent detection AUROC, (iii) incremental value of RA over Intent (joint probe / conditional AUROC), SEPARATELY from cos-independence. If best-detection layer ≠ most-independent layer, report BOTH. Script currently computes directions only; detection/joint-probe to be added. Compute ~seconds from cache.
