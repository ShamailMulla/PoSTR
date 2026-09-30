import os
import pickle
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage import color
from bsuite.environments.deep_sea import DeepSea


from .settings import STATE_SIZE
if TYPE_CHECKING:
    from ..agent.btrl import BTRL


class _DeepSeaWrapper:
    """
    Minimal gym-style wrapper around bsuite DeepSea that avoids bsuite.utils.gym_wrapper
    (which requires the legacy `gym` package, not installed in this venv).
    """
    def __init__(self, size: int, deterministic: bool = True, seed: int = 0,
                 randomize_actions: bool = True, mapping_seed: int = 0, legacy_goal: bool = False):
        # randomize_actions=True is the real DeepSea: which action index means "right"
        # is drawn per cell from mapping_seed. Train and test environments must share
        # mapping_seed, otherwise they are different tasks (bsuite draws a fresh random
        # mapping when mapping_seed is None).
        self._env = DeepSea(size, deterministic=deterministic, randomize_actions=randomize_actions,
                            seed=seed, mapping_seed=mapping_seed)
        self._size = size
        # legacy_goal reproduces the pre-2026-09 BTRL task: reaching the bottom-right cell
        # ends the episode with reward 0.99 (4 steps), instead of bsuite's +1 for moving
        # right *from* that cell on step 5. Only for reproducing old runs.
        self._legacy_goal = legacy_goal

    def reset(self) -> np.ndarray:
        ts = self._env.reset()
        return ts.observation.flatten().astype(np.float32)

    def step(self, action: int):
        ts = self._env.step(int(action))
        obs  = ts.observation.flatten().astype(np.float32)
        rew  = float(ts.reward) if ts.reward is not None else 0.0
        done = bool(ts.last())
        if self._legacy_goal and obs[-1] == 1:
            done, rew = True, 0.99
        return obs, rew, done, {}

    def action_spec(self):
        return self._env.action_spec()

    def observation_spec(self):
        return self._env.observation_spec()


class _MemoryChainWrapper:
    """
    Gym-style wrapper around bsuite MemoryChain (non-Markovian memory task).
    Observation (flattened, num_bits+2 dims): [time_to_live_frac, query, context...].
    Context is visible only at step 0, the query only at the final step — the
    agent must carry the context across the episode to answer correctly
    (terminal reward +1/-1). `env` config value = memory_length.
    """
    def __init__(self, memory_length: int, num_bits: int = 1, seed: int = 0):
        from bsuite.environments.memory_chain import MemoryChain
        self._env = MemoryChain(memory_length, num_bits, seed=seed)

    def reset(self) -> np.ndarray:
        ts = self._env.reset()
        return ts.observation.flatten().astype(np.float32)

    def step(self, action: int):
        ts = self._env.step(int(action))
        obs = ts.observation.flatten().astype(np.float32)
        rew = float(ts.reward) if ts.reward is not None else 0.0
        done = bool(ts.last())
        return obs, rew, done, {}

    def action_spec(self):
        return self._env.action_spec()

    def observation_spec(self):
        return self._env.observation_spec()


def init_env(suite: str, env: str, test: bool, deterministic: bool = True,
             randomize_actions: bool = True, mapping_seed: int = 0, legacy_goal: bool = False):
    if suite.lower() == "bsuite":
        if randomize_actions and mapping_seed is None:
            mapping_seed = 0   # never let bsuite draw an unseeded mapping (train/test would differ)
        kw = dict(deterministic=deterministic, randomize_actions=randomize_actions,
                  mapping_seed=mapping_seed, legacy_goal=legacy_goal)
        environment = _DeepSeaWrapper(int(env), seed=0, **kw)
        print('deterministic:', deterministic, '| randomize_actions:', randomize_actions,
              '| mapping_seed:', mapping_seed, '| legacy_goal:', legacy_goal)

        test_environment = None
        if test:
            test_environment = _DeepSeaWrapper(int(env), seed=1, **kw)   # same task as training

        action_space = environment.action_spec().num_values
        observation_space = int(np.prod(environment.observation_spec().shape))
    elif suite.lower() == "bsuite_memory":
        environment = _MemoryChainWrapper(int(env), seed=0)
        test_environment = _MemoryChainWrapper(int(env), seed=1) if test else None
        action_space = environment.action_spec().num_values
        observation_space = int(np.prod(environment.observation_spec().shape))
    else:
        raise NotImplementedError(f"{suite} is not available.")

    return environment, list(range(action_space)), test_environment, observation_space


def env_step(env: "Env", suite: str, action: int):
    if suite == "bsuite":
        observation, reward, done, _ = env.step(action)
        observation = observation.flatten()
        truncated = False
    elif suite == "bsuite_memory":
        observation, reward, done, _ = env.step(action)
        observation = observation.flatten()
        truncated = False

    done = done or truncated

    return observation, reward, done


def env_reset(env: "Env", suite: str):
    if suite in ("bsuite", "bsuite_memory"):
        observation = env.reset()
        observation = observation.flatten()
    return observation


def project_onehot(x: torch.Tensor) -> torch.Tensor:
    """
    v9: project continuous state logits onto the one-hot manifold (argmax).
    Used so the value and terminal networks are queried on the same input
    distribution they are trained on (buffer states are one-hot for grid
    observations). Only valid for one-hot observation spaces.
    """
    idx = x.argmax(-1, keepdim=True)
    out = torch.zeros_like(x)
    out.scatter_(-1, idx, 1.0)
    return out

def one_hot_encode(idx, obs_space):
    data = np.zeros((len(idx), obs_space))
    data[np.arange(len(idx)), idx] = 1

    return data


def compute_state_loss(
    predicted: torch.Tensor,
    true_states: torch.Tensor,
    obs_type: str,
    obs_space: int,
) -> torch.Tensor:
    """
    Dispatch state prediction loss based on observation/environment type.

    Args:
        predicted:   raw logits  [B, obs_space]
        true_states: one-hot or continuous target  [B, obs_space]
        obs_type:    "grid" | "categorical" | "continuous"
        obs_space:   flat observation dimension (N*N for a grid)

    obs_type choices:
      "grid"        — NxN one-hot grid (e.g. DeepSea).
                      MSE on logits + Smooth L1 on soft-decoded (row, col).
      "categorical" — discrete states with no spatial structure.
                      Cross-entropy (proper classification loss).
      "continuous"  — real-valued state vectors.
                      MSE directly on predicted vs true values.
    """
    if obs_type == "grid":
        return _grid_state_loss(predicted, true_states, obs_space)
    elif obs_type == "categorical":
        return _categorical_state_loss(predicted, true_states)
    elif obs_type == "continuous":
        return _continuous_state_loss(predicted, true_states)
    else:
        raise ValueError(
            f"Unknown obs_type '{obs_type}'. "
            f"Choose from: 'grid', 'categorical', 'continuous'."
        )


def _grid_state_loss(
    predicted: torch.Tensor,
    true_states: torch.Tensor,
    obs_space: int,
) -> torch.Tensor:
    """
    MSE on logits + Smooth L1 (Huber) distance in (row, col) grid space.

    Procedure:
      1. Soft-decode predicted logits → expected (row, col) via softmax.
         Differentiable: gradients flow through the softmax weighting.
      2. Hard-decode true one-hot → integer (row, col) via argmax.
      3. Smooth L1 per coordinate: Euclidean-like for small errors,
         L1 for large errors — robust early in training, no instability near zero.
    """
    mse = F.mse_loss(predicted, true_states)

    N = int(round(obs_space ** 0.5))
    idx = torch.arange(obs_space, dtype=torch.float32, device=predicted.device)
    rows = idx // N
    cols = idx % N

    pred_probs = F.softmax(predicted, dim=-1)
    pred_row = (pred_probs * rows).sum(-1)
    pred_col = (pred_probs * cols).sum(-1)

    true_idx = true_states.argmax(-1).float()
    true_row = torch.div(true_idx, N, rounding_mode='floor')
    true_col = true_idx % N

    spatial = (F.smooth_l1_loss(pred_row, true_row)
             + F.smooth_l1_loss(pred_col, true_col))
    return mse + spatial


def _categorical_state_loss(
    predicted: torch.Tensor,
    true_states: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy for discrete states with no spatial/geometric structure."""
    true_idx = true_states.argmax(-1)
    return F.cross_entropy(predicted, true_idx)


def _continuous_state_loss(
    predicted: torch.Tensor,
    true_states: torch.Tensor,
) -> torch.Tensor:
    """Plain MSE for real-valued state vectors."""
    return F.mse_loss(predicted, true_states)


def preprocess_image(image: np.array):
    """
    This function preprocesses images such that they become 64x64 grayscale images to ensure compatibility with
    PSBTRL.
    """
    if type(image) == tuple:
        image = image[0]
    if type(image) == dict:
        image = image["image"]
    if len(image.shape) == 3:
        image = color.rgb2gray(image)
    elif len(image.shape) < 2:
        return image, False
    if image.shape[0] != STATE_SIZE or image.shape[1] != STATE_SIZE:
        image = Image.fromarray(image)
        image = image.resize((STATE_SIZE, STATE_SIZE), Image.NEAREST)
    image = np.array(image)
    if np.max(image) > 1:
        image /= 255

    return image, True


def expand_image(img: torch.tensor):
    extra_dim = [1] if len(img.shape) == 3 else [1, 1]
    return img.expand(tuple(extra_dim) + tuple(img.shape))


def create_observation_batch(observations: torch.tensor, timesteps: torch.IntTensor, n_actions: int):
    """
    Create a batch of states and hidden states for all possible actions.
    """
    batch_size = observations.shape[0]
    states_x = observations.repeat(n_actions, 1, 1)
    timesteps_x = timesteps.repeat(n_actions, 1)
    actions = torch.tile(torch.arange(n_actions).reshape(-1,1),(batch_size,1))

    return states_x, actions, timesteps_x


def create_directories(env: str, algorithm: str, name: str):
    if not os.path.exists("./logdir/"):
        os.mkdir("./logdir/")
    env_folder = "./logdir/{}".format(env.replace('/', '-')+'-determinsitic')
    if not os.path.exists(env_folder):
        os.mkdir(env_folder)
    folder_name = env_folder + "/" + algorithm + "-{}".format(name)
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    number = len(os.listdir(folder_name))
    # Avoid duplicate
    while os.path.exists(folder_name + "/{}/".format(number)):
        number += 1
    logdir = folder_name + "/{}/".format(number)
    os.mkdir(logdir)
    os.mkdir(logdir + "checkpoints/")

    return logdir


def load(agent: "BTRL", load_dir: str):
    agent.model.transition_network.load_state_dict(
        torch.load(load_dir + "transition.pt")
    )
    agent.model.terminal_network.load_state_dict(torch.load(load_dir + "terminal.pt"))
    agent.value_network.load_state_dict(torch.load(load_dir + "value.pt"))
    with open(load_dir + "replay.pt".format(load_dir), "rb") as fn:
        agent.dataset.episodes = pickle.load(fn)
    print("Successfully loaded.")

    return agent


def preprocess(obs):
    return obs * 255 if len(obs.shape) > 1 else obs


def extract_episode_data(episodes: list):
    o   = torch.stack([ep["states"]      for ep in episodes])
    a   = torch.stack([ep["actions"]     for ep in episodes])
    o1  = torch.stack([ep["next_states"] for ep in episodes])
    r   = torch.stack([ep["rewards"]     for ep in episodes])
    t   = torch.stack([ep["terminals"]   for ep in episodes])
    ts  = torch.stack([ep["timestep"]    for ep in episodes])

    # uint8 tensors are Atari pixel images scaled [0, 255] → divide back to [0, 1].
    # float32 tensors are already in [0, 1] (one-hot vectors for DeepSea) — no scaling.
    scale_o  = 255.0 if o.dtype  == torch.uint8 else 1.0
    scale_o1 = 255.0 if o1.dtype == torch.uint8 else 1.0
    return o.float() / scale_o, a, o1.float() / scale_o1, r, t, ts

