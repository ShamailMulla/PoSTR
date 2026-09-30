# Representation Drift (The Moving Target Problem)

**Integration Problem #2** when connecting [[bayesformer|BayesFormer]] to the [[psdrl|PSDRL]] value network.

Source: `Raw Resources/integration_problem.txt`

---

## What It Is

Representation Drift occurs when the hidden states `H^a` produced by the [[transition-network|transition network]] change unpredictably from call to call, making the features fed to the value network **non-stationary**. The value network can never converge because its input distribution is always shifting.

---

## Root Cause

[[bayesformer|BayesFormer]] with standard `nn.Dropout` generates a different mask at every forward call. This means every call to `transition_network.predict()` produces a different set of hidden states `H^a`, even for the same input context:

```
predict(context, action=0) → mask_A → H^a_A
predict(context, action=0) → mask_B → H^a_B   # different!
predict(context, action=0) → mask_C → H^a_C   # different again!
```

The value network `V(H^a)` tries to approximate the Q-function. But its input `H^a` changes randomly every call, creating an **extreme version of the non-stationary target problem** in deep RL.

---

## Why This Is Worse for Transformers Than RNNs

[[psdrl|PSDRL's]] GRU transition model creates representations through simple recurrent operations. Transformers create **highly complex, non-linear dynamic representations via multi-head self-attention**. As the BayesFormer trains and attention maps update, the geometric structure of `H^a` changes drastically.

Each training step slightly shifts all attention weights → the input feature space to the value network is scrambled → the value network unlearns its previous progress → repeat forever.

---

## Cascade Effect

```
Training step k:
  H^a = f_θ(context)     # one representation
  V(H^a) → Q-values      # value network fits to H^a

Training step k+1:
  H^a = f_θ(context)     # DIFFERENT representation (mask changed)
  V(H^a) → Q-values      # value network must relearn completely
```

The value network effectively trains on a **new task every forward pass**. No convergence is possible.

---

## Manifestation in Experiments

Symptoms of Representation Drift:
- Value loss oscillates violently without converging
- Agent performance plateaus at near-random (e.g., reward only from random actions in DeepSea)
- Training curves show value loss getting worse over time even as transition loss improves
- Grid distance metrics (`mean_grid_dist`) remain high despite reasonable transition accuracy

---

## The Fix

The same fix as [[dithering-trap|Problem #1]]: [[episode-locked-dropout|EpisodeLockedDropout]].

By locking all dropout masks at episode start, every call to `predict()` during the episode uses the same mask → the same world hypothesis M̂ → **consistent `H^a` representations** throughout the episode.

The value network now trains on a **stable input distribution** for each episode, enabling convergence.

```python
# With EpisodeLockedDropout — FIXED
model.lock_all_dropouts(...)        # sample M̂ once
# ↓ all calls during episode use same mask
H_a = transition_network.predict(context, action)
V(H_a)  # stable input → value network can learn
```

---

## Connection to PSDRL Design

In [[psdrl|PSDRL]], the neural-linear uncertainty model represents uncertainty **only over the output layer weights** of a fixed feature map. This means `H^a` (the feature subnetwork output) is relatively stable across posterior samples — only the linear head changes.

BTRL's Transformer uncertainty spans **all layers** (Q, K, V, FFN). This is much richer but requires the episode-locking discipline to prevent drift.

---

## Links

- [[dithering-trap]] — Problem #1, shares the same root cause
- [[episode-locked-dropout]] — the fix
- [[transition-network]] — the source of H^a representations
- [[bayesformer]] — why Transformer uncertainty is multi-layer
- [[psdrl]] — why the original PSDRL design avoided this problem
