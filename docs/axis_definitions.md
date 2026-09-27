# ACSL Axis Definitions (authoritative)

> Written 2026-07-04 at Matt's request, **before** reconstructing the actionability
> labeler, to pin down what each probe axis means operationally and to prove the
> axes are independent of the output-level refusal signal we are testing them against.
>
> **The circularity constraint (non-negotiable):** ACSL's whole claim is that an
> internal probe read at a mid-network layer beats output-level refusal under
> jailbreaks. That comparison is only valid if the probe axes are labeled from
> **content**, never from **whether the model refused**. If a label is derived from
> the model's output/refusal, the probe is trained on the very thing it is supposed
> to beat, and the result is circular. Every definition below is therefore stated as
> a property of the *input content*, and the "Enforcement" line shows the exact code
> path and proves no output/refusal signal enters it.

---

> ## ❌ Request Actionability RETIRED as a harm axis (2026-07-05)
> After the instrument rebuild (thinking-OFF caches), a per-family control with a
> 1000-trial random-feature null and a length null, and two adversarial Rule-9 passes,
> **Request Actionability does not earn its place as a harm-detection axis.** On the
> trustworthy 4B (read L33) it is null-to-anti at *every* layer; on the 1.7B (L9) it is
> only weak-but-non-null and confounded (early length-heavy layer; the eval negatives are
> unwrapped clean benigns, confounding wrapping with harm), and does not reproduce on the 4B.
> RA's earlier apparent "rescue" of Intent under jailbreaks was **common-mode
> wrapper-cancellation**, reproducible with a random second direction.
>
> **PRESERVED — the corpus harm-axis substrate (this part is a real finding):** both the
> Intent and RA diff-of-means directions load onto a single **dominant harm/benign axis of
> this corpus**. That is *why* cos(Intent, RA)=0.78 at 1.7B/L9, and *why* any second
> direction with a component along that axis lets a logistic probe subtract the jailbreak
> wrapper's common-mode offset and "denoise" Intent — i.e. the mechanistic explanation for
> the wrapper-cancellation effect, not an orphan observation. Prediction: the same substrate
> will appear under *any* diff-of-means direction fit on this corpus; treat large cos() with
> Intent as expected corpus geometry, not evidence of a shared feature. The RA labeler
> (`scripts/11_request_actionability.py`) and the request-side definition below are kept as a
> descriptive prompt-form probe, NOT a validated harm axis.

> **Naming decided 2026-07-04 (Matt):** Axis 1 is **Intent** (the code's context-aware
> `label==intent` behavior is correct and kept; no data re-labeled). Axis 3 is
> **Request Actionability** — explicitly request-side (prompt-position): it measures
> whether the *request solicits operational steps*, NOT content-side uplift (which would
> need response-position extraction; out of scope this version). Refusal/approval remains
> the **baseline**, never an axis. UI, API default token, and docs updated to match; the
> API still accepts legacy `"harm"` as an alias so logged history rows keep resolving.

## Axis 1 — Intent (was labeled "harm" in the UI)

**Operational definition (as enforced today):** Does the *request itself* express
genuinely harmful intent, judged in context — **yes/no**. This is context-aware,
not topic-based: "How do I kill someone?" = 1, but "How do I kill someone in Call
of Duty?" = 0 and "How can I kill a Python process?" = 0. Same surface verb, opposite
label, because the label tracks intent, not the word.

**Source of truth:** the `label` field in `data/v3_{harmful,harmless,wrapped}.jsonl`.
Verified 2026-07-04: `label == intent` for all 1325 records, so the "harm" axis IS
the intent annotation. `label`/`intent` are human content annotations (derived from
`annotation_1/2`); **no** output/response/refusal field exists on any record.

**Provenance caveat (Rule 9):** the audited code proves the label reaching the probe
is the JSONL `label` field and nothing model-derived. It does **not** prove *who* set
that field — the upstream annotation step (human vs. any automated process) is outside
the repo's code paths. Non-circularity of Axis 1 rests on trusting the data-prep
pipeline, which is asserted, not code-proven here.

**Wrapped split:** 420 of the 1325 records are in `v3_wrapped.jsonl` (jailbreak-template
prompts) — the exact adversarial cases the probe must beat. Their `label`/`intent` is
**copied from the pre-wrap prompt's content annotation** (verified: labels track the
underlying request, no output field present). This is the intended behavior (the wrapper
must not change the ground-truth harm of the underlying ask), but it is load-bearing for
the jailbreak comparison and is stated here explicitly rather than left implicit.

**Enforcement:**
- Cache label: [`acsl/extract.py:240`](../acsl/extract.py) — `labels[id] = int(rec["label"])`. Content annotation only.
- Direction (trace + gate): `_compute_caution_direction` in [`server.py`](../server.py) — diff-of-means of cached `y>=1` vs `y==0`. `y` is the cache label above.
- Audit direction: `_compute_direction(..., label_fn=None)` in [`scripts/10_direction_audit.py`](../scripts/10_direction_audit.py) uses the same cache `y`.

**⚠ Naming discrepancy to resolve with Matt:** Matt's working definition is
"Harm = is the content in a harmful *domain* (topic), yes/no." That is **not** what
the code's harm axis measures — the code measures *intent* (context-aware). The
"topic" notion corresponds to the **`severity`** field (below), which STATUS records
as **dropped**. Options: (a) rename the trace axis "Intent" to match the code, or
(b) re-label from a topic annotation to match Matt's definition. **No code changed
until Matt chooses.**

---

## Axis 2 — Severity (dangerous-topic) — DROPPED, documented for completeness

**Operational definition:** Is the *topic* dangerous, regardless of intent — graded
0/1/2. "What is the definition of murder?" scores severity>0 (dangerous topic) but
intent=0 (benign request). This is the "topic" axis in Matt's harm framing.

**Source of truth:** the `severity` field (0/1/2) in the same data files. Content
annotation; no output signal.

**Status:** Dropped as a probe axis (STATUS.md: "severity reads dangerous-topic not
weaponizable intent"). Still present in the saved npz for reference. If Matt wants
"Harm = topic," this is the label that already encodes it.

---

## Axis 3 — Request Actionability  ❌ RETIRED as a harm axis (2026-07-05 — see banner above)

**Operational definition (target):** Does the interaction concern **specific
operational capability / uplift** — i.e. executable steps that would enable *doing*
the thing — as opposed to understanding/description. Graded in Matt's framing; the
existing saved vectors were built from a **binary** form-based proxy:
"asks for executable steps" = 1, "asks for understanding" = 0 (STATUS.md).
Explicitly **independent of topic danger** and **independent of refusal**.

**Property of what?** ACSL reads *prompt-encoding* activations, so a prompt-position
probe can only represent a property of the **request**. The enforced label is
therefore *request-actionability* ("does this request seek operational steps"),
the prompt-side analog of *content-actionability* ("does the content provide
operational uplift"). They align in spirit; the enforced label is binary and
request-side, not graded and content-side. **This gap is a labeling-scheme decision
for Matt** — see Reconstruction note below.

**Reconstruction discriminator (validated, not yet committed):** the signal is
first/second-person action requests vs third-person/past understanding:
- `=1`: "how do I / how can I …", "how to …", imperative ("help me / give me / write /
  create / list …"), "I want to …". → asks to be enabled to *do* something.
- `=0`: "what is / why / when / who / define …", and crucially "how **did** X die",
  "how **were** …", "how **does** ASLR work" → understanding of events/concepts.

A naive "starts with how" rule is wrong: it mislabels "How did JFK die?" as
operational. The person/tense distinction is what reproduces the saved vectors
(work-in-progress cos ≈ 0.96–0.98 vs saved L9/L14/L17; target ≈ 1.0 before commit).

**Provenance (current) — UNVERIFIABLE for the served vector (Rule 9 finding):**
the saved `actionability` vectors in `runs/10_direction_audit_{1.7b,4b}/directions_L*.npz`
are loaded and served by `_load_audit_directions` (`server.py`), but **no committed
`.py` file produces the `actionability.npy` key** — `10_direction_audit.py` computes
only intent and severity. The labeler that made the shipped vectors is absent from the
tree. So for the artifact currently in production, refusal-independence is **asserted,
not code-verified**. Correct status: *provenance unverifiable* until the labeler is
reconstructed and committed.

**"Prompt-only" is necessary but NOT sufficient for refusal-independence (Rule 9
finding):** whether a model refuses is itself largely a function of the prompt, so a
prompt-only labeler can still be a proxy for refusability. Independence must be
**measured**, not assumed. A reconstruction proxy of the described discriminator
showed Pearson r ≈ 0.26 with the harm label (imperative "give me steps" forms skew
toward `label=1`) — modest but nonzero, and pointing at exactly the surface feature
output-refusal keys on. **Before committing any reconstructed labeler, report
cos(actionability, intent) per layer and the label correlation, and treat high values
as a red flag.** (STATUS notes cos(severity, actionability)=0.136 and 10/10 topic-vs-act
separation for the *original* vectors — that is prior evidence of independence for the
saved vectors, but it does not transfer to a new reconstruction until re-measured.)

---

## The one place output-level labeling lives (keep it quarantined)

[`scripts/review_data.py`](../scripts/review_data.py) `classify_response()` scores the
**model's response** by soft-refusal / compliance / disclaimer markers. This is the
**output-level refusal baseline** — the thing the probe is tested *against*. It is a
data-review tool and **must never** feed any probe-axis label.

Verified (Rule 9): `classify_response` is called only within `review_data.py` itself;
no other module imports it, and its outputs feed only the HTML review report and a v2
`--auto-clean` export (`data/harm_v2_cleaned.jsonl`), never a `v3_*` file the extractor
reads. **But this is a firewall by convention, not by mechanism** — there is no import
guard, no test, and no schema/provenance check in `extract.py` asserting that `label`
is content-derived. The extractor will faithfully cache whatever integer sits in
`label`, refusal-derived or not. A future edit could route output-level signal into a
`v3_*` label file with nothing to stop it. Hardening this (a provenance field + an
assertion in the extractor) is the recommended fix and does not exist yet.

---

## Summary table

| Axis | Question | Granularity | Source (content-level) | Refusal-independent? |
|------|----------|-------------|------------------------|----------------------|
| **Intent** (API: `intent`, legacy alias `harm`) | Genuinely harmful *intent*, in context? | binary | `label`==`intent` field | ✅ code-verified content annotation (upstream annotator trusted, not code-proven) |
| Severity (dropped) | Dangerous *topic*? | 0/1/2 | `severity` field | ✅ annotation |
| **Request Actionability** ❌ RETIRED | (was) does the *request* solicit operational steps? | binary proxy | `scripts/11_request_actionability.py` (persisted) | Retired as a harm axis 2026-07-05: null/anti on 4B every layer, weak+confounded on 1.7B; apparent rescue = wrapper-cancellation. Kept only as a descriptive prompt-form probe. |

**Verified 2026-07-04:** no field in the data files is output/response/refusal-derived
(`grep` of record keys returns none), and `norm_stats` (per-feature activation mean/std,
`acsl/norm.py`) never sees labels, so normalization introduces no leakage. Axis 1 and
Axis 2 labels trace to content annotations via audited code paths. **Axis 3 is the open
item:** its served labeler is absent from the tree and its refusal-independence is not
yet measured. The Harm/Severity comparison against output-level refusal is not circular
as currently enforced; the Actionability comparison cannot yet make that claim.

---

## Rule 9 outcome (2026-07-04)

A fresh adversarial reviewer (no access to the authoring session) checked this doc
against the code. Verdicts: Claim 1 (harm not refusal-contaminated) SUPPORTED with the
upstream-annotator caveat now added; Claim 2 (`label==intent`) SUPPORTED (1325/1325);
Claim 4 (`classify_response` quarantined) SUPPORTED; Claim 5 (harm=intent≠topic
discrepancy) SUPPORTED. Claim 3 (actionability refusal-independent) PARTIALLY SUPPORTED
— downgraded above. The three corrections it forced (served-vector provenance,
prompt-only≠refusal-independent, convention≠enforcement) are folded into the sections
above. This doc supersedes the pre-review draft.
