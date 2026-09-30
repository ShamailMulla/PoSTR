# Episode-Locked Dropout

The single most important design constraint in [[btrl|BTRL]]. Implemented as `EpisodeLockedDropout` in `src/TRL/networks/transition.py`.

**It is the fix for both [[dithering-trap|Problem #1]] and [[representation-drift|Problem #2]].**

---

## The Problem with `nn.Dropout`

PyTorch's standard `nn.Dropout` generates a **new random mask on every forward call**. During rollout, the [[transition-network|transition network]] is called at every timestep. This means:

- Step t: mask = {Q drops head 2, K drops head 0, ...} → world hypothesis A
- Step t+1: mask = {Q drops head 1, K drops head 2, ...} → world hypothesis B
- Step t+2: mask = different again → world hypothesis C

The agent imagines a **different world at every step**. This is not [[thompson-sampling|Thompson Sampling]]; it's step-wise randomised execution, which causes the [[dithering-trap]].

---

## The Fix: `EpisodeLockedDropout`

```python
class EpisodeLockedDropout(nn.Module):
    """
    Two modes:
    - Unlocked (training): standard stochastic dropout — new mask each forward call
    - Locked (rollout):    a mask is sampled once and reused every forward call
    """
    def __init__(self, p: float):
        super().__init__()
        self.p = p
        self._mask = None

    def lock(self, shape: tuple, device):
        """Sample and freeze one mask for the episode."""
        self._mask = (torch.rand(shape, device=device) > self.p).float() / (1 - self.p)

    def unlock(self):
        """Return to stochastic mode."""
        self._mask = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._mask is not None and self._mask.shape == x.shape:
            return x * self._mask          # locked: reuse frozen mask
        return F.dropout(x, p=self.p, training=self.training)  # stochastic fallback
```

**Stochastic fallback:** The shape check `self._mask.shape == x.shape` is critical. Training calls use batch dimension `B=100, L=10`, while rollout uses `N_actions=2, L=2`. The dimensions never match during training, so training always uses stochastic dropout. No explicit unlock needed between training steps.

---

## All 8 Dropout Sites

Every dropout site in the [[bayesformer|BayesFormer]] architecture must be locked. Missing even one causes residual step-wise stochasticity that corrupts the temporal features:

| # | Site | Instance | Class |
|---|---|---|---|
| i | State embedding dropout | `enc.state_dropout` | `Encoder` |
| i | Positional encoding dropout | `enc.pos_dropout` | `Encoder` |
| i | Action embedding dropout | `enc.action_dropout` | `Encoder` |
| ii | Q projection dropout | `attn.q_dropout` | `MultiHeadAttention` |
| ii | K projection dropout | `attn.k_dropout` | `MultiHeadAttention` |
| ii | V projection dropout | `attn.v_dropout` | `MultiHeadAttention` |
| iii | Attention map dropout (post-softmax) | `attn.attn_map_dropout` | `MultiHeadAttention` |
| iv | FFN skip-connection dropout | `block.dropout` | `TransformerBlock` |

---

## Lock Shapes

The mask shape must match the tensor shape during rollout:
- `N` = num_actions (rollout batch dimension)
- `L` = context_length (before interleaving)
- `2L` = after state-action interleaving in `Encoder`
- `embed` = hidden_dim

| Site | Shape |
|---|---|
| State/pos/action embedding | `(N, L, embed)` |
| Q/K/V projections | `(N, 2L, heads, head_dim)` |
| Attention map | `(N, heads, 2L, 2L)` |
| FFN skip | `(N, 2L, embed)` |

---

## Usage in BTRL

```python
# Episode start (in BTRL.select_action when step == 1)
model.sample_model()
# ↳ calls transition_network.lock_all_dropouts(N_actions, context_len, device)
# Locks all 8 sites with shapes matching rollout dimensions

# During episode — all predict() calls reuse the frozen masks
next_states, rewards = transition_network.predict(...)

# After episode / before training — unlock for stochastic training
transition_network.unlock_all_dropouts()
```

---

## Links

- [[dithering-trap]] — Problem #1 that this fixes
- [[representation-drift]] — Problem #2 that this fixes
- [[thompson-sampling]] — the strategy that requires episode-consistent masks
- [[bayesformer]] — source of the multi-site dropout architecture
- [[transition-network]] — where EpisodeLockedDropout is implemented
- [[btrl]] — overall system
