# ACSL — Head Spec (two-axis, baseline → band-ensemble, ablation ladder)

Defines the probe heads for the gate. Two axes (intent, severity), built as an ablation ladder so each richer head is measured against the simpler one and kept only if it wins on **held-out** wrapper families. Reads the residual stream `h_ℓ` (z-scored per layer, train-fit stats).

> Prereq before building: confirm the 4B Stage 0 result is stable (re-run sweep with a second seed) and the read position is the same structural token in clean vs wrapped (special-token check). Build heads on a confirmed band, not a provisional one.

---

## Axes (settled)

Two **separate** heads, never one graded multi-class head:
- **Intent head** `I` — malicious framing? (benign ↔ malicious)
- **Severity head** `S` — weaponizable content? (inert ↔ dangerous)

Each trained/fit independently, each evaluated for invariance independently. The gate reads both; the payoff cell is `low-intent / high-severity → answer`.

---

## The ladder (build in order, each gated by beating the previous on held-out families)

### Rung 0 — diff-of-means, single-layer (BASELINE)
What Stage 0 already computed, promoted to a head. This is the number everything else is measured against.
```
per axis a, at the single best band layer ℓ*:
    u[a] = normalize( mean(h_norm[ℓ*] | label_a high) - mean(h_norm[ℓ*] | label_a low) )   # train-clean only
    score_a(x) = ⟨ h_norm[ℓ*](x), u[a] ⟩
    threshold τ_a set on held-out HARD-benign per over-refusal budget
```
- ℓ* = the single best layer from the band sweep (4B: ~32; re-confirm per axis — intent and severity peak at different depths).
- No training beyond the means. Zero new parameters. This is the honest floor.

### Rung 1 — trained linear, single-layer
Same one layer, but fit a logistic-regression weight vector instead of class-mean difference.
```
per axis a, at ℓ*:
    w[a], b[a] = logreg.fit( h_norm[ℓ*](train), label_a )      # class-weighted (handles imbalance)
    score_a(x) = sigmoid( w[a] · h_norm[ℓ*](x) + b[a] )
```
- Class-weight the fit (your sev-0/sev-2 skew) so the direction isn't dragged by the majority class — this is the *correct* fix for imbalance, not downsampling.
- **Gate:** must beat Rung 0 AUROC on held-out families. Usually a small honest gain; if it doesn't beat diff-of-means, keep diff-of-means (simpler).

### Rung 2 — band-ensemble (THE PROMISING ONE on 4B)
Read the **whole invariant band**, not one layer, and combine. This is the structural move that exploits the 4B's 21-layer-wide band — a wrapper must suppress many redundant layers at once.
```
per axis a, over band B = [the invariant layers for axis a]:
    option A (pooled feature): concat or mean the per-layer projections
        feat(x) = [ ⟨h_norm[ℓ](x), u[a,ℓ]⟩  for ℓ in B ]      # |B|-dim vector
        score_a(x) = logreg.fit(feat, label_a)                  # learns per-layer weights
    option B (vote): score_a(x) = aggregate({ per-layer scores }, agg=max|mean)
```
- Start with **option A pooled-logreg** (learns which band layers matter); option B (max-vote) is the cheap version — max is robust because a wrapper must push *every* layer under threshold, not just one.
- Use each layer's own `u[a,ℓ]` (the direction rotates through depth).
- **Gate:** must beat Rung 1 on held-out families. This is where the robustness payoff should show up — if ensemble doesn't beat single-layer, the band's redundancy isn't helping and you note that as a finding.

### Rung 3 — MLP, single-layer or band (OPTIONAL, gravy)
Non-linear reader. Only worth it to answer "is the linear read leaving signal on the table."
```
per axis a:
    head = MLP(in=d_model or |B|, hidden=small, out=1)          # 1 hidden layer, keep it small
    train on (h_norm[ℓ*] or band feature, label_a)
```
- Keep the hidden layer small — your data is limited, big MLP overfits.
- **Gate:** must beat the best of Rung 1/2 on held-out families. Lower priority now that scale already pushed the hard boundary to ~0.89; build only if chasing the last points.

---

## Suppression head (Stage 2 — CONDITIONAL, separate from the ladder)

Build ONLY if the Stage 1 divergence test shows intent and severity diverge under wrapping. Different target from the axes — predicts *masking*, not harm.
```
input: trajectory features [i_B, s_B, Δi_B, Δs_B, (Δi−Δs)]
target: wrapped vs genuine   (from wrapper_family)
head: small classifier; train on TRAINING wrapper families, test on HELD-OUT families
compose: if suppressed → treat intent as HIGH (distrust the masked read)
```
- This is the head that catches the suppressed-by-wrapper cases (most of the gate-misses).
- Do not build before Stage 1 confirms the divergence signature exists.

---

## Training discipline (applies to every rung)

- **Direction/weights fit on TRAINING-clean only.** Never fit on wrapped or held-out — apply the clean-trained head to wrapped data. (Fitting on wrapped is the circularity that produced the fake 0.99.)
- **Per-layer z-score** with train-fit mu/sigma, applied identically to all splits.
- **Class-weight** to handle sev-0/sev-2 imbalance — do not downsample.
- **Thresholds** set on held-out HARD-benign per over-refusal budget; report ASR and over-refusal **together**.

---

## Evaluation (the ladder's verdict)

Per axis, on **held-out wrapper families** (the 2×2, unseen-family + unseen-prompt as headline):
- AUROC clean and wrapped, delta (invariance), sub-domain split (general vs security — general is the hard boundary).
- ablation table: Rung 0 → 1 → 2 → (3), each row's held-out AUROC. Keep the simplest rung that isn't beaten.

Joint gate:
- ASR (held-out wrapped-harmful through the gate) **with** over-refusal (hard benign), bootstrap CIs.
- the 2×2 cell behavior: confirm `low-intent/high-severity` actually passes (the dual-use payoff).

**Decision rule (pre-register):** a richer head ships only if it beats the simpler one on held-out-family AUROC by more than the bootstrap CI. Otherwise keep the simpler head. Complexity must be earned on unseen attacks, not on training fit.

---

## Build checklist
- [ ] 4B Stage 0 confirmed stable (second seed) + read-position special-token check done
- [ ] Rung 0 diff-of-means heads (intent + severity) at each axis's best layer — the baseline
- [ ] Stage 1 divergence test run (gates the suppression head)
- [ ] Rung 1 trained-linear, class-weighted — compare to Rung 0 on held-out
- [ ] Rung 2 band-ensemble (pooled-logreg) — compare to Rung 1 on held-out
- [ ] Rung 3 MLP — only if chasing remaining points
- [ ] suppression head — only if Stage 1 showed divergence
- [ ] two-axis gate wired (2×2 policy), paired eval with ablation table
