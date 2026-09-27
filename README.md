# ACSL — Attention-Coupled Security Loop

A **defensive safety-evaluation harness** for frozen open-weight LLMs. ACSL reads
two linear probes from the residual stream at prompt-encoding time, **Intent**
(is the request malicious, judged in context) and **Request-Actionability** (does
the request solicit operational steps), and gates the model's action on the pair
rather than on the model's own surface refusal. The design question is whether an
*internal* harm read survives jailbreak wrapping better than output-level refusal
does, **at matched over-refusal**, and whether a second axis lets legitimate
dual-use requests through.

This is research code (architecture v3). The README states what has been
established and what has not; read that section before quoting any number.

---

## What it does

- **Residual-stream read, every layer, frozen base.** Activations `h_ℓ` at the
  last prompt token are cached to disk for all layers; the base model is never
  trained through. Reading the residual (not the attention output) is what makes
  the probe architecture-agnostic.
- **Two axes, not one score.** Diff-of-means directions (and optionally trained
  linear/MLP heads) for Intent and Request-Actionability, labeled from **prompt
  content only**, never from whether the model refused
  (`docs/axis_definitions.md` proves the non-circularity path by path).
- **2×2 gate.** high-intent/high-action → refuse or route; high-intent/low-action →
  redirect; **low-intent/high-action → answer** (the dual-use cell: a benign
  exploit explanation passes while malice is caught); low/low → answer.
- **Layer selection by kill-test invariance.** A layer is eligible only if its
  clean-vs-jailbreak-wrapped AUROC delta stays under the pre-set bound across
  **held-out wrapper families**. Clean AUROC alone picks surface-form-dependent
  layers that fail under wrapping; this is a load-bearing methodological result.
- **Paired metrics by construction.** Every catch rate / ASR number is reported
  next to over-refusal on the same prompts with bootstrap CIs, so a
  refuse-everything policy visibly fails.
- **Fail-closed steering loop.** `acsl/loop.py` reads → steers along the caution
  direction → re-reads for K iterations; if it does not converge or risk stays
  high, the safe action is forced. (The loop exists and is unit-tested; it has
  **not** been evaluated end to end on the v3 axes.)
- **Live instrument.** `server.py` + `frontend/index.html` (port 5000): token ×
  layer harm trace with 2D/3D heatmaps, logit lens, gate 2×2 view, history and
  compare, SQLite logging.

## Prior art and what is new here

Activation probes as a safety screen are prior art (e.g. Anthropic's
Constitutional Classifiers++ deploys a linear-probe screen); reading multiple
layers is routine; the read-position choice follows Zhao et al. ACSL
independently converges on the same internal harm signal with a different
instrument. What it adds:

1. Intent × Request-Actionability as **separately kill-tested** residual reads
   with an explicit dual-use answer cell.
2. Per-axis invariant-band selection by kill test over held-out wrapper families
   rather than by clean accuracy.
3. A documented **retirement under controls**: the apparent "Actionability
   rescues Intent under jailbreaks" effect is reproduced by a *random* second
   direction (1000-trial null) and is common-mode wrapper cancellation on a
   dominant corpus harm axis, not a second harm feature.

## What has been established, and what has not

Established on the corrected (thinking-OFF) caches, Qwen3-1.7B and Qwen3-4B,
fp16 on an AMD RX 6900 XT under ROCm:

- The instrument trust suite passes on both models: polarity correct, lexical
  positive control decodable, label-shuffle collapses to chance, train→held-out
  wrapper-family transfer gap ≤ 0.03 (`scripts/13_gate_validation.py`).
- Rung-0 diff-of-means heads are invariant on 12/12 held-out families
  (4B intent @ L33 clean 0.937 / wrapped 0.902; 1.7B intent @ L9 0.761 / 0.737).
- The 2×2 gate runs at the 5% over-refusal target; the dual-use cell is correct
  on 92.8% of its prompts.
- The ablation ladder (diff-of-means → linear → band → MLP) tops out near a
  43% gate catch rate at 5% over-refusal on 4B; Intent on hard wrapper families
  is the bottleneck. These ladder numbers were computed on the earlier
  thinking-ON caches and have not been re-derived on the corrected ones.
- Severity as an axis reads "dangerous topic", not weaponizable intent, and was
  dropped (a clean negative result).
- Request-Actionability was **retired** as a harm axis after the random-direction
  null (see above). Its labeler is kept as a descriptive prompt-form probe.

Not established (do not cite as results):

- The `t_inst` read-position number in `STATUS.md` has no script or run artifact
  in the repo and predates the tokenization fix.
- The steering loop has not been evaluated end to end.
- **Back-verification is open** (`CLAUDE.md`): the refusal labels used for the
  v1 paired eval were machine-classified and never human-audited, and the v1
  results ran on a DirectML backend that has not been cross-checked against
  ROCm. The repo's own rule is that no ACSL result is cited until both close.
- Corpus limits: the wrapped split used in reported runs contains no
  wrapped-*benign* prompts, so wrapped-vs-benign contrasts confound wrapping
  with harm; and about 39% of the 1.7B @ L9 Intent score is explained by prompt
  length (4B @ L33 is clean).

`STATUS.md` carries the full dated record including retracted claims;
`FINDINGS.md` is the v1 interim report (Qwen2.5-1.5B, DirectML) and is
superseded in method by v3.

## Data policy

The repo ships **only** public-benchmark-derived prompt files
(`data/xstest_safe.jsonl`, `data/xstest_unsafe.jsonl`, from XSTest).
`scripts/prepare_data.py` fetches AdvBench, Alpaca, XSTest and public in-the-wild
jailbreak templates from their canonical sources and builds the schema files.

The custom-authored security corpora and the jailbreak-wrapped sets used in the
reported v3 runs are **not distributed**. The v3 configs reference them as
`data/v3_harmful.jsonl`, `data/v3_harmless.jsonl`, `data/v3_wrapped.jsonl`; to
run v3 you supply files in the schema below. No harmful model outputs are
stored anywhere in the repo.

```json
{"id": "ex-001", "prompt": "<text>", "label": 1, "intent": 1, "severity": 2,
 "wrapper_family": "persona_override", "original_id": "ex-000"}
```

`label == intent` (0 benign, 1 malicious) is the probe target; `severity`
(0/1/2) is a topic annotation; `wrapper_family` is required on wrapped rows and
is what the held-out-family kill test splits on. Labels must come from prompt
content, never from model responses.

## Environment

Device-agnostic PyTorch; the reference stack is Docker on the pinned ROCm image
(torch 2.9.1 + ROCm 7.2.4, AMD RX 6900 XT / gfx1030). `constraints.txt` and
`scripts/setup_container.sh` keep pip from replacing the torch build. On ROCm the
GPU is device `"cuda"`. fp16 reductions overflow on this stack: all sums and
means are accumulated in fp32, which is load-bearing. The 4B model must load in
fp16 (fp32 exceeds 16 GB).

CPU-only development needs no GPU or model download:

```bash
pip install -e ".[dev]"
pytest -q                                 # hooks fire once, loop fails closed, metrics
python scripts/00_smoke.py --fake-model   # end-to-end wiring, no network
```

## Run order (v3)

Every script takes `--config configs/<name>.yaml` and writes a run manifest.
Use the `*_thinkoff` configs; they match the serving tokenization.

| step | script | what it does |
|---|---|---|
| 1 | `01_extract.py --no-enable-thinking` | cache residual `h_ℓ` at all layers, clean + wrapped splits |
| 2 | `06_kill_test.py` | per-layer diff-of-means AUROC clean vs wrapped; invariant band per axis |
| 3 | `10_direction_audit.py` | direction vectors at chosen layers; spectrum, minimal pairs, steering asymmetry |
| 4 | `11_request_actionability.py` | persisted request-side actionability labeler + independence report |
| 5 | `13_gate_validation.py` | trust suite: polarity, positive control, shuffle, shortcut R², cross-family |
| 6 | `14_per_family_control.py` | per-family rescue test against random-direction and length nulls |
| — | `server.py` | live trace / gate UI on the cached directions |

The v1 pipeline (`00`–`08`: head sweep, faithfulness, loop calibration,
pre-registration, kill test, paired eval, ablations) is retained and documented
in `FINDINGS.md`; it targets Qwen2.5-1.5B and the attention-output tap that v3
replaced with the residual read.

## Server

```bash
./launch_server.sh      # container on port 5000, waits for model load
./kill_server.sh
```

Requests are accepted from loopback and the Docker bridge gateway; add clients
with `ACSL_ALLOWED_IPS="a.b.c.d,e.f.g.h"`. There is no authentication beyond
the allowlist; do not expose the port beyond a trusted LAN.

## Layout

```
acsl/            hooks, extract, head, directions, norm, loop, policy, prereg, tokenization, eval/
scripts/         00–08 v1 pipeline; 09–14 v3 analyses; prepare_data, wrap/review/reframe tooling
configs/         default / hard (v1), v3 / v3_4b and *_thinkoff (v3)
docs/            architecture_v3, build_steps_v3, head_spec_v3, axis_definitions (authoritative)
tests/           hooks, fail-closed loop, metrics
server.py, frontend/index.html, launch_server.sh, kill_server.sh
STATUS.md        dated session record incl. corrections and retired claims
FINDINGS.md      v1 interim report
CLAUDE.md        working rules: tiered verification, Rule 9 adversarial review, back-verification order
```

## Process

Result-bearing claims go through a fresh-context adversarial review before they
are cited (Rule 9 in `CLAUDE.md`): argue the null first, reproduce at least one
headline number, name at least three attack vectors and one omitted confound.
Three claims in this repo were corrected or retired by that process and the
record is kept in `STATUS.md` and `docs/axis_definitions.md`.

## License

Apache-2.0.
