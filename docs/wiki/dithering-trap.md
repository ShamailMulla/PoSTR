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

---

## Scope — what the trap does and does not rule out (added 2026-09-26)

The argument above is about **per-step** resampling: a new world hypothesis on every
forward call. It does **not** rule out resampling on a schedule, and it was
over-applied when it was used to set aside [[psdrl|PSDRL]]'s every-`m`-steps sampling.

| Schedule | Dithering? | Status |
|---|---|---|
| Per step (new mask / sample every forward call) | **Yes** — this page | never |
| Per episode | No | PSRL's schedule; the default for episodic DeepSea |
| Every `m` steps with `m ≥` episode length (PSDRL: `m = update_freq`, 1e3 in its Atari config) | No — one model spans several episodes | a heuristic for long / continuing episodes; PSDRL needs it because refitting its value net is expensive |
| Growing / data-doubling schedule | No | theoretically grounded for continuing tasks (Thompson Sampling with Dynamic Episodes, Ouyang et al. 2017 — verify) |
| Every `m` steps with `m <` episode length | Mild — the model switches mid-episode | avoid |

PSDRL's `m` is not about simulating a changing environment — the environment is
assumed stationary. It is a practical approximation to per-episode PSRL when episodes
are long and the value network is costly.

## Known issue in the v3+ agent (found 2026-09-26)

The neural-linear agent mixes two sampling schedules:

1. `sample_model()` draws the **wide** exploration sample (`explore=True`) at step 1
   of every episode, for acting.
2. `NeuralLinearHead.update_posteriors()` ends with `self.sample()`, drawing a new
   **narrow** sample every `update_freq` (100) steps. The value network is trained
   under that sample, and because the call can land mid-episode (≈1 episode in 20
   in DeepSea-5), it also **replaces the acting model mid-episode**.

So the value network is trained for one sampled model and used to act under a
different one for the next ~20 episodes; PSDRL keeps these the same by using one
sample per `m` steps for both. This mismatch may contribute to the inert value
network (untested). [[planning-in-sampled-model]] removes the issue: the planner
uses the same per-episode sample it acts with, and the mid-episode redraw goes.

## Links

- [[planning-in-sampled-model]] — sampling schedules in the PSRL theory (§7)
- [[representation-drift]] — Problem #2, co-caused by the same issue
- [[episode-locked-dropout]] — the fix
- [[thompson-sampling]] — the guarantee that requires episode-consistency
- [[transition-network]] — where the dropout lives
- [[deepsea]] — the environment where this matters most
