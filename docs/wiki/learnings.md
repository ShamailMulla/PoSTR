# Learnings — What Works and What Doesn't

Session-by-session log. Newest entries at the top. Each entry covers discoveries, validated approaches, and dead ends.

---

## Session: 2026-07-13/14 — BayesFormer vs PSDRL Uncertainty; Return to Neural-Linear

The deciding question of this phase: after swapping GRU → Transformer, which of the
two uncertainty representations to keep — [[bayesformer|BayesFormer]] dropout or
[[psdrl|PSDRL]] neural-linear? See [[failure-modes]] for the full narrative.

### ✅ What Works

**Neural-linear (BLR) head as the posterior ([[neural-linear-head]])**
- Data-built precision `Λ = βΦᵀΦ + αI` on the transformer's penultimate features
  → genuine self-widening (validated: 870× epistemic variance on a goal region whose
  data was removed; BLR mean accuracy matches the deterministic head).
- The *only* apples-to-apples posterior comparison to PSDRL, and the basis for the
  attention/uncertainty interpretability story. This is the current version.

**Deciding that BayesFormer dropout is the wrong tool**
- Dropout's predictive spread is a function of the mask rate, not the data, so it
  cannot self-widen where data is scarce — an architecture-level limitation that no
  tuning fixes. This is the principled reason to return to PSDRL's neural-linear head.

### ❌ What Didn't Work / Gotchas

**Reading uncertainty off dropout**
- Sampled dropout hypotheses agreed ~98–100 % even when wrong — overconfident and
  not data-coupled. Confirmed the dropout posterior is unfit for directed exploration.

**Whole-buffer feature forward OOMs**
- Computing BLR features over the entire replay at once OOMs at ~1e4 transitions with
  several seeds sharing a GPU. Fix: chunk feature maps (512) + `expandable_segments`.

### 🔬 Open Questions
- Does the current neural-linear version hold DeepSea-5 across seeds over 10⁶ steps?
  (multi-seed validation in progress)
- POMDP direction (memory_chain, longer context) — next once the above is confirmed.

---

## Session: 2026-06-25 — Loss Function Redesign + Wiki

### ✅ What Works

**Environment-aware loss dispatch via `compute_state_loss()`**
- Single function in `TRL/common/utils.py`, called from TRL training, notebook training cells, and `early_stopping_train()` in `notebook_utils.py`
- `obs_type="grid"` for DeepSea: MSE on logits + Smooth L1 on soft-decoded (row, col) coordinates
- Spatial term validated: adjacent-cell prediction loss (400.54) < far-cell prediction loss (407.04) — correct gradient signal
- Soft-decode via softmax is differentiable; argmax is not — critical for backprop through spatial term

**Smooth L1 (Huber) over true Euclidean for spatial loss**
- True Euclidean `sqrt(dx² + dy²)` has gradient instability near distance=0
- Smooth L1: Euclidean-like for small errors, L1 for large — best of both
- Gradients confirmed non-zero in smoke test (norm 0.235)

**`obs_type` in config rather than hardcoded**
- Notebooks read `config['transition']['obs_type']` rather than hardcoding `'grid'`
- Future env types: `"categorical"` (cross-entropy), `"continuous"` (MSE)

**Removing `grid_distance_loss` from `notebook_utils.py`**
- Was dead code (never called), different implementation (true Euclidean expected distance)
- Replaced by `compute_state_loss` — single canonical implementation

### ❌ What Didn't Work / Gotchas

**MSE on raw logits vs one-hot is spatially blind**
- This was the original `state_loss = F.mse_loss(predicted, next_states.squeeze(1))`
- Penalises wrong-cell predictions equally regardless of grid distance
- Misaligns training loss with evaluation metric (`mean_grid_dist`)

**`sed -i 's/^transition:$/transition:\n  obs_type...'` on Linux**
- The `\n` in sed's replacement string doesn't expand on GNU sed in bash heredoc
- Fix: use Python string replacement instead

**Notebook cell `source` is a list of characters after `json.dump`**
- Multi-line string replacements in notebook source need to account for character-list format
- Using `''.join(cell['source'])` then `list(new_src)` works reliably
- Indentation of continuation lines in multiline function calls must be preserved manually

---

## Session: [Earlier] — EpisodeLockedDropout Integration

### ✅ What Works

**`EpisodeLockedDropout` with shape-based stochastic fallback**
- Training uses `(B=100, L=10)` dimensions; rollout uses `(N_actions=2, L=2)`
- Dimensions never match → training always uses stochastic dropout automatically
- No explicit unlock needed between training iterations
- Unlock only needed between episodes during rollout

**Locking all 8 sites (not just Q/K/V)**
- Missing any site leaves residual step-wise stochasticity
- Attention map dropout (post-softmax, site iii) is particularly important — corrupts temporal features if unlocked
- FFN skip-connection dropout (site iv) also corrupts H^a representations

### ❌ What Didn't Work

**Freezing only attention head projections (Q, K, V)**
- Seemed sufficient but allowed attention map dropout and FFN masks to flicker
- Still caused Representation Drift — hidden states H^a were non-stationary
- Must lock ALL sites

**Using `nn.Dropout` for rollout**
- Each forward call resamples mask → [[dithering-trap|Dithering Trap]]
- Value network cannot converge — [[representation-drift|Representation Drift]]

**GRU-to-Transformer direct swap without rethinking uncertainty representation**
- PSDRL's GRU forward model uses neural-linear BLR (uncertainty only in output layer)
- Simply replacing with BayesFormer without episode-locking breaks Thompson Sampling entirely
- Must understand that Transformer dropout = multi-layer posterior, not just output-layer

---

## Session: [Earlier] — Initial BTRL Integration

### ✅ What Works

**Context window approach for Transformer input**
- `[(timestep, state, action)]` for last `context_length` steps
- State and action embeddings interleaved → 2L sequence
- Enables Transformer self-attention to capture temporal dependencies

**BayesFormer dropout probability 0.2**
- Consistent with paper's recommendation (0.05–0.2 range)
- Too high (>0.3) → too much information lost per forward pass
- Too low (<0.05) → insufficient uncertainty diversity across hypotheses

**DeepSea-5 as primary test environment**
- N=5 (25 states) is hard enough to require Thompson Sampling but tractable enough to debug
- Deterministic variant preferred for diagnosis (removes environment noise from metrics)

### ❌ What Didn't Work

**Large hidden_dim without sufficient data**
- hidden_dim=512 with only 100 training transitions → severe overfitting
- 1000 transitions + 100 epochs is a reasonable baseline for ablations
- Optuna sweep validated: hidden_dim=512, lr=1e-4 performs best at scale

**Training on batched sequential data without windowed TBPTT**
- Computing full-sequence gradients per update causes memory issues for long sequences
- `window_length=3` TBPTT (truncated backprop through time) works well

---

## Template for New Sessions

```
## Session: YYYY-MM-DD — [Theme]

### ✅ What Works
- Finding 1 with explanation
- Finding 2 with explanation

### ❌ What Didn't Work / Gotchas
- Dead end 1 with reason
- Gotcha 1 with fix

### 🔬 Open Questions
- Question 1
- Question 2
```

---

## Links

- [[btrl]] — the project
- [[episode-locked-dropout]] — key fix
- [[loss-functions]] — most recent change
- [[dissertation-feedback]] — examiner guidance informing priorities
