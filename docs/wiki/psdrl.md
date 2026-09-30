# PSDRL — Posterior Sampling for Deep Reinforcement Learning

**Paper:** "Posterior Sampling for Deep Reinforcement Learning"
**Authors:** Remo Sasso, Michelangelo Conserva, Paulo Rauber (Queen Mary University of London)
**arXiv:** 2305.00477 | ICML 2023

---

## Problem it Solves

Deep RL agents are notoriously **sample inefficient** — they require huge amounts of trial-and-error interaction to learn. Model-based methods improve this by building a world model to plan with. But scaling Bayesian/posterior sampling approaches to non-tabular environments had been elusive.

PSDRL is the **first truly scalable** approximation of Posterior Sampling for RL (PSRL) that retains its model-based essence.

---

## Algorithm Overview

PSDRL represents knowledge about the environment as a **distribution over world models** and repeats:

1. **Sample** one model M̂ from the posterior
2. **Plan** optimally under M̂ to find policy π*
3. **Act** under π* for one episode
4. **Update** the posterior with collected data

This is [[thompson-sampling|Thompson Sampling]] applied to model-based RL.

```
Algorithm 1: PSDRL
D ← empty replay buffer (capacity C)
s_0 ← initial state, h_0 ← 0
for each t in {0, ..., T-1}:
    if t mod m == 0:
        W̃ ← sample from posteriors         # Thompson Sampling
        f_θ̃ ← forward model from W̃         # sample one world model
        ψ ← update value network under f_θ̃
    z_t ← E_χ(s_t)                         # encode state
    a_t ← π̃(z_t, h_t)                      # greedy under sampled model
    r_t, s_{t+1}, δ ← env.step(a_t)
    (h_{t+1}) ← f_θ̃(z_t, a_t, h_t)
    D ← D ∪ {(s_t, a_t, r_t, s_{t+1}, δ)}
    # update χ, θ, η (every m steps)
```

---

## Architecture Components

### 1. Autoencoder (χ)
- Encodes high-dimensional state `s_t` → low-dimensional latent `z_t = E_χ(s_t)`
- Decoder `D_χ(z_t) → ŝ_t` for reconstruction
- Loss: `L_AE(χ) = (1/BL) Σ ||D_χ(E_χ(s)) - s||²`

### 2. Forward Model (θ) — the transition model
- GRU-based: `(ẑ_{t+1}, r̂_t, h_{t+1}) = f_θ(z_t, a_t, h_t)`
- Predicts next latent state, reward, and new hidden state
- Loss: `L_F(θ) = (1/BL) Σ [||ẑ_{t+1} - z_{t+1}||² + (r̂_t - r_t)²]`

### 3. Uncertainty Model (neural-linear)
- Bayesian linear regression over the last hidden layer features
- Maintains a Gaussian posterior over output layer weights W
- Posterior: `N(w_j | μ_j, Σ_j)` where `Σ_j⁻¹ = σ_j⁻² I + σ⁻² Φ^T Φ`
- Enables efficient closed-form posterior updates

### 4. Termination Model (η)
- Predicts episode termination probability `δ̂ ∈ [0,1]`

### 5. Value Network (ψ)
- MLP approximating Q-function under sampled model f_θ̃
- Loss: `L_V(ψ) = (1/BL) Σ (V_ψ(z, h) - max_a[r̂ + γv̂_{t+1}])²`
- Policy: `π̃(z_t, h_t) = argmax_a [r̂_t^(a) + γv̂_{t+1}^(a)]`

---

## Key Design Choices

**Latent space:** Planning in z-space is cheaper than raw pixel space. Enables scaling to complex environments.

**Neural-linear uncertainty:** Closed-form Bayesian updates over only the output layer parameters — computationally cheap, no MCMC needed.

**Value function carries information across sampled models:** The value network retains knowledge from previous episodes, unlike naive PSRL which would discard it.

**Recency bias:** Replay buffer limits capacity, so posterior updates are based on recent transitions only. Acknowledged limitation.

---

## BTRL Extension

[[btrl|BTRL]] replaces PSDRL's GRU-based forward model with a [[bayesformer|BayesFormer]] Transformer that:
- Takes a context window `[(s_{t-L}, a_{t-L}), ..., (s_t, a_t)]` instead of a single `(z_t, a_t, h_t)`
- Uses dropout as the uncertainty representation instead of neural-linear BLR
- Samples world hypotheses by [[episode-locked-dropout|locking dropout masks]] at episode start

The motivation: Transformers' self-attention over trajectory context enables better world-model learning in sparse-reward, deep-exploration tasks.

---

## Links

- [[thompson-sampling]] — theoretical basis
- [[btrl]] — thesis extension replacing GRU with BayesFormer
- [[transition-network]] — the BTRL Transformer transition model
- [[bayesformer]] — the uncertainty mechanism used in BTRL
- [[deepsea]] — benchmark environment for experiments
