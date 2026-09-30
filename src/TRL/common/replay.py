import numpy as np
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..common.logger import Logger
from ..common.utils import extract_episode_data, preprocess
from ..common.settings import STATE_SIZE


class Dataset:
    def __init__(self, logger: "Logger", config: dict, time_limit: int,context_length:int,device: str,
        state_size : int, seed: int = None
    ):

        self.cum_rew = 0
        self.add_idx = 0
        self.n_samples = 0
        self.max_ep_len = 0
        self.min_ep_len = 0
        self.episode_add_idx = 0
        self.total_num_transitions = 0
        self.replace = False
        self.episodes = []
        self.logger = logger
        self.device = device
        self.random_state = np.random.RandomState(seed)
        self.context_length = context_length
        self.capacity = config["capacity"]
        self.batch_size = config["batch_size"]
        self.sequence_len = config["sequence_length"]

        # Protected partition: episodes containing reward (stored tanh-squashed,
        # so goal transitions are ~0.757) are COPIED into a separate FIFO ring
        # that only goal episodes compete for. Age-based eviction in the main
        # ring purges all reward signal whenever a performance dip outlasts one
        # buffer turnover; the protected ring guarantees both a presence floor
        # and a sampling-share floor for goal data. protected_fraction = 0
        # disables the ring and reproduces the original behaviour exactly.
        self.protected_fraction = float(config.get("protected_fraction", 0.0))
        self.protected_capacity = self.protected_fraction * float(self.capacity)
        self.goal_episodes = []
        self.goal_add_idx = 0
        self.goal_n_samples = 0
        self.goal_replace = False

        self.tmp_episode = {
            "states": torch.zeros(
                (time_limit, self.context_length, state_size),
                dtype=torch.float32,
                device=self.device,
            ),
            "actions": torch.zeros((time_limit + 1, 1), dtype=torch.int32, device=self.device),
            "next_states": torch.zeros(
                (time_limit, state_size),
                dtype=torch.float32,
                device=self.device,
            ),
            "rewards": torch.zeros((time_limit + 1, 1), device=self.device),
            "terminals": torch.zeros((time_limit + 1, 1), device=self.device),
            "timestep": torch.zeros((time_limit, self.context_length), dtype=int, device=self.device)
        }

    def add_data(self,
        new_state: np.array, new_action: torch.IntTensor, new_next_state: np.array, new_reward: int, new_terminal: bool,
                 episode_step: int):
        """
        Add new transition to a temporary episode until a terminal state is reached. Then, append episode to replay
        buffer or replace it with the episode at self.episode_add_idx if capacity is reached.
        """
        # new_state is the context-window trajectory: shape (1, ctx_len, state_size)
        # strip the leading batch dim before storing into (ctx_len, state_size) slot
        state_to_store = new_state.squeeze(0) if (
            isinstance(new_state, torch.Tensor) and new_state.dim() == 3
        ) else new_state
        self.tmp_episode["states"][self.add_idx] = state_to_store.float()
        self.tmp_episode["actions"][self.add_idx] = new_action.detach().clone()
        self.tmp_episode["next_states"][self.add_idx] = torch.from_numpy(new_next_state).float()
        self.tmp_episode["rewards"][self.add_idx] = torch.tensor(np.tanh(new_reward))
        
        self.tmp_episode["terminals"][self.add_idx] = torch.tensor(new_terminal)
        self.tmp_episode["timestep"][self.add_idx] = episode_step

        self.add_idx += 1
        self.n_samples += 1
        self.cum_rew += new_reward
        # print('tmp_episode actions:',self.tmp_episode['actions'])
        if new_terminal:
            new_episode = {
                "states": self.tmp_episode["states"][:self.add_idx].clone(),
                "actions": self.tmp_episode["actions"][:self.add_idx].clone(),
                "next_states": self.tmp_episode["next_states"][:self.add_idx].clone(),
                "rewards": self.tmp_episode["rewards"][:self.add_idx].clone(),
                "terminals": self.tmp_episode["terminals"][:self.add_idx].clone(),
                "timestep": self.tmp_episode["timestep"][:self.add_idx].clone(),
                "cum_rew": self.cum_rew,
            }
            # print('Episode:',new_episode)
            if self.n_samples > self.capacity:
                self.n_samples = 0
                self.episode_add_idx = 0
                self.replace = True

            if self.replace and self.episode_add_idx < len(self.episodes):
                self.episodes[self.episode_add_idx] = new_episode
            else:
                self.episodes.append(new_episode)
            self.episode_add_idx += 1

            # Copy (not divert) reward-bearing episodes into the protected
            # ring: the main ring stays an unbiased picture of recent
            # experience, the protected ring is an insurance archive that a
            # performance dip cannot purge. FIFO within the ring keeps the
            # most recent goal-reaching behaviour.
            if self.protected_capacity > 0 and bool(
                (new_episode["rewards"] > 0.5).any()
            ):
                self.goal_n_samples += len(new_episode["rewards"])
                if self.goal_replace and self.goal_add_idx < len(self.goal_episodes):
                    self.goal_episodes[self.goal_add_idx] = new_episode
                else:
                    self.goal_episodes.append(new_episode)
                self.goal_add_idx += 1
                if self.goal_n_samples > self.protected_capacity:
                    self.goal_n_samples = 0
                    self.goal_add_idx = 0
                    self.goal_replace = True
                self.logger.add_scalars(
                    ["Data/Protected Episodes", "Data/Protected Transitions"],
                    [
                        len(self.goal_episodes),
                        sum(len(ep["rewards"]) for ep in self.goal_episodes),
                    ],
                )

            (
                self.min_ep_len,
                self.max_ep_len,
                self.total_num_transitions,
            ) = self.logger.add_replay_statistics(self.episodes + self.goal_episodes)

            self.add_idx = 0
            self.cum_rew = 0
        # else:
        #     print('Not new_terminal')

    def sample_sequences(self):
        """
        Sample B sequences of length L, where L is set to the current minimum episode length if no episodes exist that
        last longer than L. A sequence is sampled by sampling a random episode, then sampling a random valid
        timestep within that episode, and finally extracting the sequence episode[timestep, ..., timestep+L].
        """
        batches = []
        # Sample uniformly over the union of the main and protected rings.
        pool = self.episodes + self.goal_episodes
        sequence_length = (
            self.sequence_len
            if self.max_ep_len > self.sequence_len
            else self.min_ep_len
        )
        for _ in range(self.batch_size):
            ep_len = 0
            while ep_len < sequence_length:
                ep = self.random_state.randint(0, len(pool))
                ep_len = len(pool[ep]["states"])
            timestep = self.random_state.randint(0, ep_len - sequence_length + 1)

            batches.append(
                {
                    "states": pool[ep]["states"][
                        timestep: timestep + sequence_length
                    ],
                    "actions": pool[ep]["actions"][
                        timestep: timestep + sequence_length
                    ],
                    "next_states": pool[ep]["next_states"][
                        timestep: timestep + sequence_length
                    ],
                    "rewards": pool[ep]["rewards"][
                        timestep: timestep + sequence_length
                    ],
                    "terminals": pool[ep]["terminals"][
                        timestep: timestep + sequence_length
                    ],
                    "timestep": pool[ep]["timestep"][
                        timestep: timestep + sequence_length
                    ],
                }
            )
        return extract_episode_data(batches)

