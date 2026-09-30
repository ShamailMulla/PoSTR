# The Dithering Trap

**Integration Problem #1** when connecting [[bayesformer|BayesFormer]] to the [[psdrl|PSDRL]] loop.

Source: `Raw Resources/integration_problem.txt`

---

## What It Is

The Dithering Trap occurs when the dropout mask in the [[transition-network|transition network]] is **resampled at every timestep** during rollout instead of being locked for the entire episode.

The result: the agent's **imagined world model changes every single step**. Step t imagines the world going left; step t+1 imagines it going right. The agent dithers — executing a local Brownian random walk instead of a coordinated exploration strategy.

---

## Why It Destroys [[thompson-sampling|Thompson Sampling]]

Thompson Sampling's guarantee of deep exploration relies on **episode-wise consistency**:

> "At the start of an episode, the agent samples one concrete hypothesis of the world (M̂), locks it in, and acts optimally according to that specific hypothesis for the entire episode."

Step-wise mask resampling violates this. Instead of executing one coherent strategy under M̂, the agent executes:
- Step 0: hypothesis A → go right
- Step 1: hypothesis B → go left  
- Step 2: hypothesis C → go right
- ...

In [[deepsea|DeepSea]], reaching the reward requires going diagonally right for N consecutive steps. Step-wise dithering makes this exponentially unlikely.

---

## What Causes It (Technical)

PyTorch `nn.Dropout` generates a fresh random mask on **every `.forward()` call**. In the BTRL rollout loop, `transition_network.predict()` is called at every timestep for every possible action. Without [[episode-locked-dropout|EpisodeLockedDropout]], every call samples a new mask:

```python
# Without EpisodeLockedDropout — BROKEN
for timestep in episode:
    for action in possible_actions:
        next_state = transition_network.predict(context, action)
        # ↑ nn.Dropout samples a NEW mask here — different world hypothesis!
```

---

## Cascade Effect

1. Agent imagines inconsistent worlds step-to-step
2. Policy cannot commit to a long sequence of actions
3. Exploration degrades to local random walk (Brownian motion / dithering)
4. Value network receives inconsistent targets (see [[representation-drift]])
5. Learning stalls entirely

---

## The Fix

[[episode-locked-dropout|EpisodeLockedDropout]]:

```python
# Episode start
transition_network.lock_all_dropouts(N_actions, context_len, device)
# → samples ONE mask per site, freezes it

# All subsequent predict() calls during episode use the SAME mask
for timestep in episode:
    for action in possible_actions:
        next_state = transition_network.predict(context, action)
        # ↑ reuses frozen mask → same world hypothesis M̂ at every step
```

---

## Analogy

> "Imagine trying to navigate a maze where the walls randomly teleport to new positions every second. You'd never make coherent progress. Thompson Sampling requires the walls to stay fixed for the duration of your walk."

---

## Links

- [[representation-drift]] — Problem #2, co-caused by the same issue
- [[episode-locked-dropout]] — the fix
- [[thompson-sampling]] — the guarantee that requires episode-consistency
- [[transition-network]] — where the dropout lives
- [[deepsea]] — the environment where this matters most
