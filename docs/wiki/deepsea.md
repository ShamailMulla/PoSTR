# DeepSea Environment

The benchmark environment used in the [[btrl|BTRL]] thesis. Part of Google DeepMind's **bsuite** (Behaviour Suite for Reinforcement Learning).

---

## Description

DeepSea is a classic **hard exploration** benchmark — sparse reward, requires coordinated long-horizon strategies.

- **State space:** N×N grid, one-hot encoded → `obs_space = N²` (25 states for N=5)
- **Action space:** {0, 1} — effectively left/right (with environment-specific semantics)
- **Transitions:** Diagonal movement — going "right" moves diagonally down-right; going "left" moves diagonally down-left and resets column to 0
- **Reward:** 0.99 only when reaching state (N-1, N-1) — the **bottom-right corner**
- **Episode length:** N steps before automatic reset
- **Difficulty:** scales as `O(2^N)` — reaching the goal by random action has probability `0.5^N`

---

## Why It Tests Exploration

In DeepSea-5, the agent must take action "right" at **every single one of 5 timesteps** to reach the reward. A random agent succeeds with probability `(0.5)^5 = 3.125%`. As N grows, this becomes astronomically unlikely without a coordinated exploration strategy.

The optimal exploration strategy is exactly what [[thompson-sampling|Thompson Sampling]] provides: commit to one world hypothesis (e.g., "always go right") for an entire episode. If that hypothesis is correct, the agent discovers the reward. If wrong, it collects data to update the posterior.

An ε-greedy or random exploration agent effectively performs a random walk on the grid, making it unable to reach the bottom-right consistently.

---

## Grid Structure

```
N=5 example:
(0,0) (0,1) (0,2) (0,3) (0,4)
(1,0) (1,1) (1,2) (1,3) (1,4)
(2,0) (2,1) (2,2) (2,3) (2,4)
(3,0) (3,1) (3,2) (3,3) (3,4)
(4,0) (4,1) (4,2) (4,3) (4,4) ← GOAL (reward=0.99)

Flat index: row*N + col  (e.g., (2,3) = 2*5+3 = 13)
```

**Row-boundary artefact:** Flat indices 4 and 5 have flat-diff=1, but in grid space: (0,4) to (1,0) = distance √17 ≈ 4.1. This is why [[loss-functions|loss functions]] must use grid-space (row, col) distances, not flat-index differences.

---

## Environment Variants

| Config key | Variant | Notes |
|---|---|---|
| `deterministic: true` | Deterministic | Action always has intended effect |
| `deterministic: false` | Stochastic | Action occasionally flips |
| `env: "3"` | Depth 3 (9 states) | Easier, used for sanity checks |
| `env: "5"` | Depth 5 (25 states) | Main experimental setting |
| `env: "10"` | Depth 10 (100 states) | Harder, not in main experiments |

---

## State Encoding

States are **one-hot encoded** flat vectors:
```python
obs_space = N * N      # 25 for N=5
state = np.zeros(obs_space)
state[row * N + col] = 1.0
```

The `env_step()` function in `TRL/common/utils.py` handles the special terminal state:
```python
if observation[-1] == 1:   # last element = terminal indicator
    done = True
    reward = 0.99
```

---

## Evaluation Metrics

**`mean_grid_dist`** — primary metric for [[transition-network|transition network]] accuracy:
```
mean Euclidean distance in (row, col) space
between predicted and true next state
across all test (state, action) pairs
```

`0.0` = perfect prediction. Handles row-boundary artefacts correctly.

**`exact_match`** — fraction of (state, action) pairs where the consensus prediction equals the true next state.

**`grid_variance`** — `var(pred_rows) + var(pred_cols)` across hypothesis samples. Measures posterior spread. Should approach 0 for a deterministic environment with good epistemic calibration.

---

## Why Row-Boundary Matters for Loss

In a 5×5 grid, the transition from state 4 → state 5 has flat-index difference 1 but Euclidean grid distance ≈ 4.12 (moving from col=4 to col=0 on the next row). A spatially-blind MSE loss treats this as "nearly correct". The [[loss-functions|grid state loss]] with soft-decoded coordinates correctly captures this.

---

## Links

- [[btrl]] — the agent tested on DeepSea
- [[thompson-sampling]] — why DeepSea requires Thompson Sampling
- [[loss-functions]] — why spatial loss matters for grid environments
- [[transition-network]] — the model that predicts DeepSea transitions
