import sys
import numpy as np
import torch
import torch.nn as nn


class BayesianTransformer(nn.Module):
    def __init__(self, config: dict, states: int, actions: list, context_length:int,
                 transition_network: torch.nn.Module,
                 terminal_network: torch.nn.Module,
                 device=torch.device,
                 neural_linear_head=None):
        super().__init__()

        self.device = device
        self.transition_network = transition_network
        self.terminal_network = terminal_network
        # v3: when a neural-linear head is supplied it owns the posterior over
        # world models (sample_model / predict route through it); otherwise the
        # BayesFormer dropout mechanism is used (v1/v2).
        self.head = neural_linear_head
        self.actions = torch.tensor(actions, dtype=torch.int32).to(self.device)
        self.num_actions = len(self.actions)
        self.state_size = states
        self.context_length = context_length
        self.trajectory = torch.zeros((1,self.transition_network.context_length,  self.transition_network.states), device=self.device)
        self.time_transition = torch.zeros((1,self.transition_network.context_length), dtype=int, device=self.device)

    def sample_model(self):
        """
        Sample one world hypothesis M̂ for this episode (Thompson Sampling step).
        v3: draw a weight matrix from the neural-linear posterior. v1/v2: lock every
        BayesFormer dropout site to a fixed mask reused all episode.
        Called by BTRL at step == 1 of each episode.
        """
        if self.head is not None:
            # acting draws the WIDE (exploration) sample; value training uses the
            # true-posterior sample re-drawn in head.update_posteriors().
            self.head.sample(explore=True)
        else:
            self.transition_network.lock_all_dropouts(
                self.num_actions, self.context_length, self.device
            )

    def predict(self, step: int, current_state: np.array, simulation=False):
        """
        Simulate one timestep using the current sampled model(s).
        """
        # print('\nTransformer RL Agent')
        # print('model.trajectory',self.trajectory.shape)
        if not simulation:
            # agent is interaction live with the environment
            # print('step:', step)
            if step == 1:
                self.trajectory = torch.zeros_like(self.trajectory).clone()
                self.time_transition = torch.zeros_like(self.time_transition).clone()

            # updating transition information with current state input
            self.trajectory = torch.cat([
                self.trajectory[:,1:,:],
                torch.tensor([current_state], dtype=torch.float32, device=self.device).unsqueeze(0)
            ],dim=1)
            # print('self.time_transition (before)', self.time_transition)
            self.time_transition = torch.cat([
                self.time_transition[:,1:],
                torch.tensor([step], dtype=int, device=self.device).unsqueeze(0)
            ],dim=1)
            # print('self.time_transition',self.time_transition)
            # adding current_state as a batch for each action
            observations = torch.cat([self.trajectory for _ in self.actions], dim=0)
            timestep = torch.cat([self.time_transition for _ in self.actions], dim=0)
            actions = self.actions.reshape(-1,1)
        else:
            observations = torch.cat([current_state for _ in self.actions], dim=0)
            timestep = torch.cat([step for _ in self.actions], dim=0)
            actions = torch.tensor([[
                action for _ in range(len(current_state))
            ] for action in self.actions], dtype=int, device=self.device).reshape(-1,1)

        if self.head is not None:
            # v3: predict via the sampled neural-linear weights on transformer features
            next_state_preds, return_preds, terminals = self.head.predict(timestep, observations, actions)
        else:
            next_state_preds, return_preds, _ = self.transition_network.forward(timestep, observations, actions)
            terminals = self.terminal_network.predict(next_state_preds)

        return next_state_preds, return_preds.reshape(-1, 1), terminals
