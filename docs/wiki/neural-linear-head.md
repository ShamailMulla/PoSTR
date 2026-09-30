# Neural-Linear Head — the Return to PSDRL Uncertainty

## What this is

The **current version** of BTRL. After [[bayesformer|BayesFormer]] dropout was set
aside (see [[failure-modes]]), the posterior over world models is represented the
same way [[psdrl|PSDRL]] does it: a **Bayesian linear regression (BLR) head** on the
transformer's penultimate features. This isolates the *one* architectural change
this thesis makes — recurrent (GRU) → attention (Transformer) world model — while
keeping PSDRL's proven, data-coupled uncertainty mechanism. It is therefore the only
apples-to-apples comparison to the baseline.
Config: `configs/config_v3_neural_linear.yaml` (`algorithm.neural_linear: true`).

## Why dropout was set aside for this

A posterior must be **wide where data is scarce and tight where data is dense**.
[[bayesformer|Dropout]] cannot represent that — its spread is a function of the mask
rate, not the data. The BLR precision `Λ = βΦᵀΦ + αI` is assembled from the features
actually seen, so it self-widens. This is an architecture-level distinction; see
[[failure-modes]] for the full argument.

## What the BLR head does

The transformer encoder produces features `Φ = transition.features()`
(post-ReLU penultimate of `predict_state`, dim = `hidden_dim`). A Bayesian linear
regression maps `[Φ, 1] → [next-state logits (25), reward]`:

```
Λ  = β · ΦᵀΦ + α · I          # data-built precision  (β = noise_variance, α = blr_coefficient)
Σ  = Λ⁻¹                       # Cholesky inverse
μ  = β · Σ · Φᵀ Y              # posterior mean
W ~ N(μ, Σ)                    # one Thompson sample per episode
```

- `sample_model()` → `head.sample()` (draw one `W` per episode, true-posterior
  Thompson sampling with `transition_temp = reward_temp = 1`).
- `predict()` → `head.predict()` (`W·Φ`).
- `btrl.update()` calls `head.update_posteriors(dataset)` after the transformer
  gradient step and before value training.
- Dropout is demoted to `p_seq = p_attn = 0` (so the [[dithering-trap|dithering trap]]
  cannot occur).

Implementation: `TRL/bayes/neural_linear_head.py`. Feature maps are computed in
chunks of 512 transitions to bound GPU memory (whole-buffer forward OOMs at ~1e4
transitions when several seeds share a GPU).

## Validation (on a checkpoint that solved DeepSea-5)

`validate_neural_linear.py`, fitting the BLR on 80 % of the checkpoint's real
replay features:

| Test | Result | Verdict |
|---|---|---|
| **1 — mean accuracy** — BLR posterior-mean next-state grid distance vs the deterministic head | 1.74 vs 1.76 | ✅ PASS (linear head on penultimate is expressive enough) |
| **2 — self-widening** — remove all goal transitions, refit, compare epistemic variance of the (removed) goal-approach transition vs a common one | goal region **870× more uncertain** | ✅ PASS (posterior widens where data was removed) |

The head captures data-coupled uncertainty as designed — the property dropout lacked.

## Status

Full DeepSea-5 evaluation of the current version is in progress (multi-seed, 10⁶
steps). This is the configuration the thesis reports as the working BTRL agent.

## Links

- [[failure-modes]] — the version history and why dropout was set aside
- [[psdrl]] — the baseline whose neural-linear model this ports
- [[bayesformer]] — the dropout posterior this replaces
- [[value-network]] — the MLP the head's samples feed
