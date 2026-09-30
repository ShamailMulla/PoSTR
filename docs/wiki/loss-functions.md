# Loss Functions

Environment-aware state transition loss for the [[transition-network|transition network]].
**Canonical location:** `src/TRL/common/utils.py` — `compute_state_loss()`

---

## Motivation

The original `state_loss = F.mse_loss(predicted, one_hot_target)` is **spatially blind**:
- Predicting the adjacent grid cell incurs the same loss as predicting the opposite corner
- For [[deepsea|DeepSea]] (5×5 grid), this means predicting index 1 (1 cell away) vs index 24 (√32 ≈ 5.66 cells away) from index 0 receive identical gradient signal
- Misaligns training objective with evaluation metric (`mean_grid_dist` — Euclidean distance)

---

## The Unified API

```python
compute_state_loss(
    predicted: torch.Tensor,   # raw logits  [B, obs_space]
    true_states: torch.Tensor, # one-hot or continuous  [B, obs_space]
    obs_type: str,             # "grid" | "categorical" | "continuous"
    obs_space: int,            # flat observation dimension
) -> torch.Tensor
```

`obs_type` is read from config key `transition.obs_type` everywhere.

---

## Loss Functions by obs_type

### `"grid"` — NxN one-hot grid (DeepSea, mazes)

Two components:
1. **MSE on logits** — keeps predictions magnitude-reasonable, penalises incorrect one-hot
2. **Smooth L1 on soft-decoded (row, col)** — spatially-aware, Euclidean-like for small errors, L1 for large errors

```python
def _grid_state_loss(predicted, true_states, obs_space):
    mse = F.mse_loss(predicted, true_states)

    N = int(round(obs_space ** 0.5))  # grid side length
    idx = torch.arange(obs_space, dtype=torch.float32, device=predicted.device)
    rows = idx // N
    cols = idx % N

    # Soft-decode: differentiable expected (row, col) from predicted distribution
    pred_probs = F.softmax(predicted, dim=-1)
    pred_row = (pred_probs * rows).sum(-1)
    pred_col = (pred_probs * cols).sum(-1)

    # Hard-decode true state
    true_idx = true_states.argmax(-1).float()
    true_row = torch.div(true_idx, N, rounding_mode='floor')
    true_col = true_idx % N

    spatial = F.smooth_l1_loss(pred_row, true_row) + F.smooth_l1_loss(pred_col, true_col)
    return mse + spatial
```

**Why Smooth L1 (Huber) over true Euclidean?**
- True Euclidean `sqrt(dx² + dy²)` has infinite gradient at distance=0 → training instability
- Smooth L1: behaves like L2 (Euclidean-like) for small errors, L1 for large errors
- Robust early in training when predictions are completely wrong

**Why soft-decode via softmax?**
- `argmax` is not differentiable — gradients cannot flow through it
- Softmax-weighted expected coordinate is fully differentiable
- The model learns "nearby cells should have higher probability than far cells"

### `"categorical"` — discrete states without spatial structure

```python
def _categorical_state_loss(predicted, true_states):
    true_idx = true_states.argmax(-1)
    return F.cross_entropy(predicted, true_idx)
```

Cross-entropy is the correct loss for classification problems. MSE on one-hot can be used but has issues with logit magnitude.

### `"continuous"` — real-valued state vectors

```python
def _continuous_state_loss(predicted, true_states):
    return F.mse_loss(predicted, true_states)
```

Plain MSE for regression over state dimensions.

---

## Reward Loss

Reward is always scalar regression — **always MSE**, regardless of `obs_type`:

```python
reward_loss = F.mse_loss(return_preds, r[:, idx].unsqueeze(1))
```

`TM_REWARD_LOSS_F = torch.nn.MSELoss()` is still in `settings.py` for reference.

---

## Configuration

```yaml
# All configs under src/configs/*.yaml
transition:
  obs_type: "grid"   # "grid" | "categorical" | "continuous"
```

---

## Where It's Called

| Location | Purpose |
|---|---|
| `TRL/training/transition.py` line 74 | Main training loop (actual experiments) |
| `notebook_utils.py` — `early_stopping_train()` | Optuna hyperparameter search training |
| Notebook `test1_...ipynb` — all ~12 training cells | Manual experiments / ablations |

All three call `compute_state_loss()` from `TRL.common.utils` with `obs_type=config['transition']['obs_type']`. There is **one definition, called everywhere**.

---

## Spatial Loss Validation

```python
# Adjacent vs far cell — spatial term correctly differentiates:
# True state: cell 0 = (row=0, col=0)
# Predict cell 1 = (row=0, col=1) → grid dist = 1
# Predict cell 24 = (row=4, col=4) → grid dist = √32 ≈ 5.66

loss_adjacent  # 400.54   (lower)
loss_far       # 407.04   (higher — spatial term adds ~6.5 extra)
```

---

## Environment Decision Table

| Suite | Environment | `obs_type` | Reason |
|---|---|---|---|
| bsuite | DeepSea-N | `"grid"` | N×N one-hot, spatial transitions |
| bsuite | (future categorical) | `"categorical"` | discrete, non-spatial |
| gym / dm_control | continuous control | `"continuous"` | real-valued state vectors |

---

## Links

- [[transition-network]] — where the loss is applied during training
- [[deepsea]] — why spatial loss matters (grid structure)
- [[btrl]] — the overall system
