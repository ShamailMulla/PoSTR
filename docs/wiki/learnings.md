# Learnings — What Works and What Doesn't

Session-by-session log. Newest entries at the top. Each entry covers discoveries, validated approaches, and dead ends.

---

## Session: 2026-09-25/26 — Recovery, rerun, and the planning proposal

Context: the thesis folder and Claude transcripts were lost; code and wiki were
rebuilt from backups into `recovered_thesis/`, and PoSTR `postr_det` (5 seeds × 1M)
was rerun with MLflow tracking and per-cycle tracing (1 in 20 update cycles).

### ✅ What Works
- **Discovery reproduces:** first treasure within 25 steps on every seed (3/5 on the
  exact same step as July).
- **The floor reproduces:** all seeds end at 11.5–13.8% solve rate over the last 100k
  (July: 11–13%).
- **Tracing localises collapses:** each collapse starts at a feature-drift spike
  (seed 3 at 529→531k: drift 0.198 vs ~1e-4; posterior distinct states 4→1).

### ❌ What Didn't Work / Gotchas
- **"Flat value net causes the failure" is wrong:** V is constant even during 99%
  phases. The policy is the BLR reward margin.
- **"The floor is an attractor" is wrong:** seed 3 escaped the floor at ~520k and
  re-solved to 99% before collapsing again.
- **July seed 1 was an outlier:** its 800k-step plateau did not recur (rerun seed 1
  never learned; 1 of 10 runs across both). PoSTR's July mean-regret advantage over
  PSDRL depended on it — use medians and discovery/floor results.
- **Sampling-schedule hybrid in v3+:** mid-episode posterior redraw; value net trained
  and used under different samples ([[dithering-trap]] § Known issue).
- **Benchmark:** runs use `randomize_actions=False`, where "always right" is optimal
  everywhere, so DeepSea cannot separate planning from an action bias.
- **Infrastructure:** a lid-close suspend kills CUDA runs (Xid 31) and can leave CUDA
  unusable until reboot; `btrl.py` then silently falls back to CPU. Launch scripts now
  check CUDA, hold a sleep inhibitor, and write 20k-step resume checkpoints.
- **Markov check (2026-09-26):** the transformer's next-state prediction depends on the
  previous state in 25–50% of cases on healthy checkpoints — spurious, since DeepSea is
  Markov. Supports Option 1 in [[planning-in-sampled-model]].
- **Feature collapse:** seed 1 (whole run) and seed 3 (≤500k) have constant transformer
  features; predictions are worse than the majority class. Healthy-feature seeds still
  sit at the floor with ~0.001 reward margins.
- **Reward model is fine; propagation is not (2026-09-26):** on healthy checkpoints
  the BLR reward model is near-exact (R² ≈ 1, goal margin +0.7, learns the −0.002 move
  cost). With an inert value net, one-step greedy then prefers *left* everywhere except
  the goal cell. See [[planning-in-sampled-model]] §11.6.
- **Task mismatch:** PoSTR trains on non-randomized DeepSea with an `env_step` override
  (episode ends with 0.99 on reaching the bottom-right cell, 4 steps); the PSDRL
  baseline trains on standard randomized DeepSea. PoSTR-vs-PSDRL numbers compare
  different tasks. ([[planning-in-sampled-model]] §9)
- **Randomized DeepSea comparison (2026-09-28/30, `RANDOMIZED_COMPARISON_REPORT.md`):**
  PSDRL (prior 1e-3) learns on 4/5 seeds and holds its 88% ceiling (policy_noise 0.05 →
  0.975⁵); PoSTR stays at the random rate (~3%) on all 5, median regret 192.9k vs 54.4k.
  PoSTR still finds the treasure first (median 1,001 vs 6,395 steps) but never keeps it.
  Value net flat on 4/5 seeds — prediction P1 of [[planning-in-sampled-model]] confirmed.
- **Reward-loss shape bug — fixed 2026-09-26:** `transition.py:82` (and the same pattern
  in `btrl.py`'s freeze-check validation loss) compared `[100,1]` with `[100,1,1]`,
  broadcasting the loss over all 100×100 prediction/target pairs. Present in every run
  before the fix. Now `view_as(return_preds)`.
- **Switched to randomized DeepSea (2026-09-26)** in both codebases; also fixed PSDRL's
  train/test envs having different random action mappings. July PSDRL test numbers
  are on a different task from its training. ([[planning-in-sampled-model]] §9)

### 🔬 Open Questions
- Does exact planning in the posterior sample + fixed-φ epochs retain the solution on
  randomized DeepSea? ([[planning-in-sampled-model]] §10)
- Does the transformer's prediction actually depend on history? (Markov check, §11)
- Is the collapse caused by the φ jump, or do both follow something earlier? Trace
  every cycle around a collapse to order the events.

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
