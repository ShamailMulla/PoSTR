# Transition Network

The core [[bayesformer|BayesFormer]] Transformer that learns the world model in [[btrl|BTRL]].
**File:** `src/TRL/networks/transition.py`

---

## Role

Given a context window of `(timestep, state, action)` tuples, predict:
- **Next state** `ŝ_{t+1}` — a probability distribution over all `obs_space` states
- **Return-to-go (reward)** `r̂` — scalar reward prediction

During rollout, it is called once per action per timestep to imagine outcomes under the current world hypothesis M̂.

---

## Architecture

```
Input: [(t_{t-L+1}, s_{t-L+1}, a_{t-L+1}), ..., (t_t, s_t, a_t)]
        context_length = L tuples

Encoder:
  ├── embed_state:  Linear(obs_space → hidden_dim) + state_dropout
  ├── embed_pos:    Embedding(max_steps, hidden_dim) + pos_dropout  
  ├── embed_action: Embedding(num_actions, hidden_dim) + action_dropout
  │   [states and actions interleaved → 2L sequence]
  └── N × TransformerBlock:
        ├── MultiHeadAttention (with Q/K/V dropouts + attn_map dropout)
        ├── LayerNorm + skip connection
        ├── FFN (Linear → ReLU → Linear)
        └── dropout (FFN skip)

Prediction Heads:
  ├── predict_state:  Flatten → Linear → ReLU → Linear(hidden_dim*2L → obs_space)
  └── predict_rtg:    Flatten → Linear → ReLU → Linear(hidden_dim*2L → 1)

Output: (next_state_logits [B, obs_space], return_preds [B, 1], attention_scores)
```

---

## Key Classes

### `EpisodeLockedDropout`
Custom dropout replacement supporting locked (rollout) and stochastic (training) modes.
See [[episode-locked-dropout]] for full details.

### `MultiHeadAttention`
```python
# 4 dropout sites per attention block:
self.q_dropout       = EpisodeLockedDropout(p)   # Q projections
self.k_dropout       = EpisodeLockedDropout(p)   # K projections
self.v_dropout       = EpisodeLockedDropout(p)   # V projections
self.attn_map_dropout = EpisodeLockedDropout(p)  # attention weights post-softmax
```
Attention: `softmax(Q·K^T / √d) · V` — standard scaled dot-product attention.

### `TransformerBlock`
```python
self.dropout = EpisodeLockedDropout(p)  # FFN skip-connection
```

### `Encoder`
Embeds states, positional encodings, and actions independently; interleaves them into a `2L` sequence; runs through stacked `TransformerBlock`s.
```python
self.state_dropout  = EpisodeLockedDropout(p)
self.pos_dropout    = EpisodeLockedDropout(p)
self.action_dropout = EpisodeLockedDropout(p)
```

### `Network` (the top-level class)
Wraps `Encoder` + prediction heads. Key attributes:
- `self.states` — obs_space (flat observation dimension)
- `self.obs_type` — "grid" / "categorical" / "continuous" (from config)
- `self.loss = 0` — accumulated windowed loss during training

Key methods:
- `forward(timesteps, observations, actions)` — training forward pass (stochastic dropout)
- `predict(timesteps, observations, actions)` — rollout (locked dropout, no grad)
- `lock_all_dropouts(N, seq_len, device)` — samples and freezes all 8 dropout masks
- `unlock_all_dropouts()` — returns to stochastic mode

---

## Training Loss

State prediction loss dispatches based on `obs_type` (see [[loss-functions]]):

```python
# In TRL/training/transition.py
transition_s1_loss = compute_state_loss(
    next_state_preds, next_state,
    obs_type=self.obs_type,    # "grid" for DeepSea
    obs_space=self.obs_space,  # 25 for DeepSea-5
)
reward_loss = F.mse_loss(return_preds, r[:, idx].unsqueeze(1))
```

For `obs_type="grid"`: MSE on logits + Smooth L1 on soft-decoded (row, col) coordinates.

---

## State Representation

For [[deepsea|DeepSea]], states are **one-hot encoded** flat vectors of length `obs_space = N²`. The flat index `s` maps to grid position `(s // N, s % N)`.

The `predict_state` head outputs **raw logits** — no final activation. During training, loss is computed directly against one-hot targets. During evaluation, `argmax(logits)` gives the predicted state index.

---

## Config

```yaml
transition:
  obs_type: "grid"          # loss function dispatch
  training_iterations: 20
  hidden_dim: 32            # swept 32-512 via Optuna
  learning_rate: 1e-2
  window_length: 3
  context_length: 2
  num_encoder_layers: 2
  heads: 2
  forward_expansion: 4
```

---

## Links

- [[bayesformer]] — the architecture this implements
- [[episode-locked-dropout]] — the custom dropout class
- [[loss-functions]] — training loss details
- [[btrl]] — the overall agent
- [[dithering-trap]] — what happens without EpisodeLockedDropout
- [[representation-drift]] — the second failure mode
