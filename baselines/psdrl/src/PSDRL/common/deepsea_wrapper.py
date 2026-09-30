from types import SimpleNamespace

import numpy as np
from bsuite.environments import deep_sea, memory_chain


class DeepSeaGymEnv:
    """Wraps bsuite DeepSea as a gym-5-tuple interface for use with main.py."""

    def __init__(self, size: int = 5, deterministic: bool = True, seed: int = 0):
        self._env = deep_sea.DeepSea(size=size, deterministic=deterministic, seed=seed)
        self.action_space = SimpleNamespace(n=2)
        self.observation_space = SimpleNamespace(shape=(size * size,))

    def reset(self):
        ts = self._env.reset()
        obs = ts.observation.flatten().astype(np.float32)
        return obs, {}

    def step(self, action):
        ts = self._env.step(int(action))
        obs = ts.observation.flatten().astype(np.float32)
        return obs, float(ts.reward), bool(ts.last()), False, {}


class MemoryChainGymEnv:
    """
    Wraps bsuite MemoryChain (non-Markovian memory task) as the same gym-5-tuple
    interface. Observation (num_bits+2,): [time_to_live_frac, query, context...];
    context visible only at step 0, query only at the final step, terminal
    reward +1/-1 for repeating the queried context bit.
    """

    def __init__(self, memory_length: int = 5, num_bits: int = 1, seed: int = 0):
        self._env = memory_chain.MemoryChain(memory_length, num_bits, seed=seed)
        self.action_space = SimpleNamespace(n=2)
        self.observation_space = SimpleNamespace(shape=(num_bits + 2,))

    def reset(self):
        ts = self._env.reset()
        return ts.observation.flatten().astype(np.float32), {}

    def step(self, action):
        ts = self._env.step(int(action))
        obs = ts.observation.flatten().astype(np.float32)
        rew = float(ts.reward) if ts.reward is not None else 0.0
        return obs, rew, bool(ts.last()), False, {}
