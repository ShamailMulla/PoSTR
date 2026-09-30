# Thompson Sampling (Posterior Sampling)

Thompson Sampling is the exploration strategy at the core of both [[psdrl|PSDRL]] and [[btrl|BTRL]]. It is the mechanism that enables **deep, coordinated exploration** in environments with sparse rewards.

---

## The Core Idea

Instead of maintaining a single belief about the world and adding an explicit exploration bonus, Thompson Sampling maintains a **distribution over world models** and acts as follows:

1. At the start of each episode, **sample one concrete world model M̂** from the posterior
2. **Commit** to this hypothesis for the entire episode — act optimally under M̂
3. Collect data, **update the posterior** distribution
4. Repeat

The key intuition: by committing to a single hypothesis per episode, the agent executes **long, coordinated action sequences** — exactly what is required to reach the reward in deep-exploration environments like [[deepsea|DeepSea]].

---

## Why Episode-Wise Commitment Matters

Consider [[deepsea|DeepSea]] depth=5. The optimal strategy is to go diagonal-right every step for 5 steps. With step-wise randomness (random action at each step), the probability of hitting the reward by chance is `0.5^5 = 3%`. 

With Thompson Sampling: if M̂ correctly models the environment, the agent executes the optimal 5-step sequence every episode under that hypothesis. Even if M̂ is occasionally wrong, different hypotheses explore different corners of the state space systematically.

**Step-wise randomised execution** (what happens without [[episode-locked-dropout]]) is:
- Not Thompson Sampling
- Equivalent to a random walk / Brownian motion
- Unable to discover rewards that require coordinated long sequences

---

## Mathematical Guarantee

For the tabular RL setting, Thompson Sampling achieves **Bayes-optimal exploration** — the posterior contracts at the same rate as if we had an oracle telling us the true environment. This bound has been proven for contextual bandits and finite MDPs.

The intuition for why it works:
- Initially, many M̂ samples are drawn → diverse exploration across the state space
- As data accumulates, the posterior concentrates → agent exploits regions where reward was found
- Exploration naturally decreases as uncertainty decreases

---

## Implementation in BTRL

In [[btrl|BTRL]], the distribution over world models is represented by the distribution over **dropout masks** in the [[transition-network|BayesFormer transition network]]:

```python
# Episode start — sample M̂
agent.model.sample_model()
# ↑ calls transition_network.lock_all_dropouts(N_actions, context_len, device)
# Each unique mask configuration = one hypothesis about the world

# During episode — act under M̂ (masks stay locked)
next_state_preds, reward_preds = transition_network.predict(context, actions)
# ↑ all calls use the same locked mask → same world hypothesis M̂

# Episode end — masks are unlocked for training (stochastic dropout)
transition_network.unlock_all_dropouts()
```

The [[episode-locked-dropout|EpisodeLockedDropout]] class is what makes this possible.

---

## Contrast with Other Exploration Strategies

| Strategy | Mechanism | Deep Exploration? | Sample Efficient? |
|---|---|---|---|
| ε-greedy | Random action with prob ε | No (local) | No |
| UCB / Optimism | Bonus for uncertain states | Yes (tabular) | Moderate |
| Intrinsic motivation | Curiosity / count bonus | Sometimes | Moderate |
| **Thompson Sampling** | Commit to sampled hypothesis | **Yes** | **Yes** |
| Randomised value functions | Distribution over Q | Yes (similar) | Yes |

---

## Links

- [[psdrl]] — the RL algorithm that scales Thompson Sampling to deep environments
- [[btrl]] — thesis project using BayesFormer for the sampled world model
- [[episode-locked-dropout]] — the mechanism that implements Thompson Sampling in BTRL
- [[dithering-trap]] — what happens without proper Thompson Sampling
- [[bayesformer]] — the uncertainty model used for sampling
- [[deepsea]] — the test environment that requires deep exploration
