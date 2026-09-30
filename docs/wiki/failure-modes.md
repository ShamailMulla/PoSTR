# Version History & Failure Modes

> The narrative of the project, and the failures found at each stage. After
> replacing PSDRL's GRU world model with a Transformer, there were **two options
> for representing Bayesian uncertainty over world models** — and we tested both:
>
> 1. **[[bayesformer|BayesFormer]] dropout** — dropout masks as an approximate
>    posterior (one mask = one hypothesis M̂).
> 2. **[[psdrl|PSDRL]] neural-linear** — a Bayesian linear regression head on the
>    encoder features (uncertainty in the output layer, precision built from data).

---

## The arc

| Version | What it is | Outcome |
|---|---|---|
| **v1** | Transformer + BayesFormer dropout — initial implementation | An implementation error (the [[dithering-trap|dithering trap]]) — dropout masks were resampled every rollout step, which destroys Thompson Sampling. Found only after extensive testing and reading through the code. Also caused [[representation-drift]]. |
| **v2** | Dithering bug fixed | [[episode-locked-dropout]] locks one mask per episode. Thompson Sampling now behaves correctly. The agent can learn — but does not hold DeepSea-5 across seeds. |
| **v3** | BayesFormer evaluated properly | **Conclusion: BayesFormer dropout is the wrong tool for the posterior.** The dropout "posterior" cannot self-widen (see below) — its uncertainty is not coupled to the data, so exploration is not directed where data is scarce. |
| **current** | Return to PSDRL neural-linear | Option 2: a [[neural-linear-head|Bayesian linear regression head]] on the transformer features, whose precision `Λ = βΦᵀΦ + αI` is built from the data. This is the scientifically clean version — it isolates the *one* architectural change (recurrent → attention world model) while keeping PSDRL's proven uncertainty mechanism. |

---

## Why BayesFormer dropout doesn't work (the deciding argument)

A posterior over world models must be **wide where data is scarce and tight where
data is dense** — that is what keeps Thompson-Sampling exploration directed toward
the unknown, and what lets the model recover if it drifts.

- **Dropout cannot do this.** Its predictive spread is roughly the same everywhere,
  regardless of how much data a state has — the "uncertainty" is a property of the
  mask rate, not of the data. Measured directly: the dropout posterior gave near-zero,
  non-data-coupled spread.
- **Neural-linear can.** Its precision is `Λ = βΦᵀΦ + αI`, assembled from the
  features actually seen. Validated on a solved checkpoint: remove all goal
  transitions, refit, and the goal region becomes **870× more uncertain** — exactly
  the self-widening dropout lacks.

This is an *architecture-level* reason (independent of any single run), which is why
it is decisive: no amount of tuning changes what dropout represents. Hence the return
to PSDRL's neural-linear head.

---

## Failure modes found along the way

| Failure | Version | Symptom | How diagnosed | Status |
|---|---|---|---|---|
| [[dithering-trap]] | v1 | World hypothesis flips every step; can't commit to an exploration strategy | Traced dropout mask resampling per forward call in rollout | ✅ Fixed — [[episode-locked-dropout]] |
| [[representation-drift]] | v1 | Value net never converges | Hidden states `H^a` non-stationary because masks resampled each call | ✅ Fixed — [[episode-locked-dropout]] |
| World-model collapse | v1 | Never learns under a tiny replay buffer | Checkpoint diagnostics: transition net predicts the buffer's marginal mode regardless of input; 98–100 % agreement across 50 sampled hypotheses | ✅ Fixed — `replay.capacity` 100 → 10⁴ |
| Dropout can't self-widen | v2/v3 | Learns then loses DeepSea-5 across seeds | Posterior spread ~equal on scarce vs dense states — not data-coupled | ➡️ Decided the return to [[neural-linear-head|PSDRL neural-linear]] |

---

## Methodology toolbox (reusable)

- `diagnose_checkpoint.py` — per-checkpoint transition/value/posterior probes.
- `validate_neural_linear.py` — BLR mean-accuracy + self-widening tests.
- Hypothesis-agreement across N sampled world models — detects posterior collapse.
- Grid-distance (not argmax accuracy) — the spatially-aware world-model metric
  ([[loss-functions]], [[optuna-objectives]]).

## Links

- [[neural-linear-head]] — the current version (PSDRL uncertainty on Transformer features)
- [[bayesformer]] — the dropout posterior that was tested and set aside
- [[dithering-trap]] · [[representation-drift]] · [[episode-locked-dropout]] — the v1 bugs and their fix
- [[learnings]] — session-by-session log
