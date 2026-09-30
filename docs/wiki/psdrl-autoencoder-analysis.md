# Why PSDRL Uses an Autoencoder — and Why We Remove It for DeepSea

## Why the CNN autoencoder exists in PSDRL

PSDRL was designed for **Atari** benchmark environments, where observations are 64×64 grayscale video frames. The raw observation space is 4,096-dimensional (64×64 pixels), making it intractable for the downstream models:

| Component | Problem without AE |
|---|---|
| GRU transition model | Would need input/hidden dim ≫ 4,096 — millions of parameters for a single GRU cell |
| BLR (Bayesian Linear Regression) | Fits a weight matrix of shape (embed\_dim+1) × latent\_dim. At raw pixel scale, the Cholesky inversion of the (Φᵀ Φ + λI) matrix becomes O(4096³) ≈ **68 billion** operations |
| Value network | 4,096-dim input per state makes the MLP impractically large |

The **CNN autoencoder** solves this by:
1. Compressing each 64×64 frame → **embed\_dim = 1,536** continuous features (a 2.7× compression)
2. Providing a smooth, differentiable latent space where the GRU can learn transition dynamics
3. Decoupling visual perception (AE) from world-model learning (GRU + BLR)

The encoder architecture (4 Conv2D layers, stride 2) progressively halves spatial resolution: 64→30→13→5→2, then flattens to 1,536 features. The decoder mirrors this with ConvTranspose2D layers.

## Why we bypass it for DeepSea 5×5

DeepSea observations are **5×5 binary matrices** (one-hot: exactly one `1` in the entire grid). Flattened, this is a **25-dimensional one-hot vector** — already in the most compact possible lossless form.

| Property | Atari frame | DeepSea 5×5 |
|---|---|---|
| Raw obs dimensions | 4,096 (64×64 pixels) | **25** (5×5 one-hot) |
| Info content | ~8-12 bits per pixel, mostly redundant | 1 bit of true information (which cell) |
| Compression needed? | Yes — aggressive (2.7×) | **No** |
| AE reconstruction loss meaningful? | Yes — pixel fidelity matters | Trivially near-zero even with random weights |

Bypassing the AE means `embed_dim = obs_dim = 25`. The GRU transition model and BLR operate directly on the raw observation. This:
- Eliminates ~7,709 AE parameters (and the associated training cost)
- Preserves exact positional information without lossy compression
- Avoids the risk of embedding collapse (AE learns to ignore fine-grained position differences)

The PSDRL paper itself acknowledges this: *"If you wish to test on environments with vectorial observations, you can either implement a different architecture for the autoencoder or **remove the autoencoder altogether**."* (README)

## Architectural comparison

```
Atari PSDRL (with AE):
  obs [64×64×1]  →  CNN Encoder  →  embed [1536]
                                         ↓
                                    GRU Cell → latent [gru_dim + hidden_dim]
                                         ↓
                                    BLR → next_embed [1536], reward [1]

DeepSea PSDRL (no AE — this thesis):
  obs [25]  ─────────────────────→  embed [25]  (identity)
                                         ↓
                                    GRU Cell → latent [gru_dim + hidden_dim]
                                         ↓
                                    BLR → next_obs [25], reward [1]
```

## Impact on hyperparameter search

Removing the AE changes the dimension that downstream networks see:

| Parameter | Atari | DeepSea (this work) |
|---|---|---|
| embed\_dim | 1,536 (tuned AE output) | **25 (fixed)** |
| BLR input dim | gru\_dim + hidden\_dim | same |
| BLR output dim | embed\_dim + 1 = 1,537 | **26** |
| Value network input | embed\_dim + gru\_dim | **25 + gru\_dim** |

The representation notebook (`param_tuning/optuna_representation_hparam_search.ipynb`) is therefore **not used for DeepSea**. It remains available for future vision-based environments where an AE is needed.

## BLR priors (algorithm section of config)

The BLR update uses two priors from `config.yaml`:

```yaml
algorithm:
  reward_prior:     1e-3   # prior precision on reward prediction weights
  transition_prior: 1e-3   # prior precision on state transition weights
```

These scale the posterior covariance:
```
transition_cov = Phi * transition_prior
reward_cov     = Phi * reward_prior
```

where `Phi = inv(ΦᵀΦ · noise_var + λI)` is the precision matrix from the BLR update.

**Smaller prior** → wider posterior → more diverse Thompson Sampling samples → more exploration.  
**Larger prior** → tighter posterior → samples close to the mean → more exploitation.

Optimal values found by hyperparameter search for DeepSea 5×5:
- `transition_prior = 0.01`
- `reward_prior = 0.1`

The reward prior is an order of magnitude larger than the transition prior, reflecting that reward prediction in DeepSea is a simpler function (almost always −0.002 per step, +1 at the goal) compared to state transitions.
