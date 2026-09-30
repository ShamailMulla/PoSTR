from copy import deepcopy

import torch

from ..bayes.transformer_rl_agent import BayesianTransformer
from ..common.replay import Dataset
from ..common.settings import TP_THRESHOLD
from ..common.utils import project_onehot


class PolicyTrainer:
    def __init__(
        self,
        config: dict,
        context_length: int,
        value_network: torch.nn.Module,
        device: str,
        batch_size: int,
        actions: torch.tensor
    ):
        self.target_net = None

        self.device = device
        self.actions = actions
        self.context_length = context_length
        self.batch_size = batch_size
        self.value_network = value_network

        self.discount = config["discount"]
        # v9 fix: bootstrap-target inputs projected to one-hot (see btrl.py)
        self.onehot_inputs = config.get("onehot_inputs", False)
        self.target_update_freq = config["target_update_freq"]
        self.training_iterations = config["training_iterations"]

        self.num_actions = len(actions)

    def train_iter(self, inputs: torch.tensor, targets: torch.tensor):
        self.value_network.optimizer.zero_grad()

        outputs = self.value_network.forward(inputs)
        self.value_network.loss = self.value_network.loss_function(outputs, targets)

        self.value_network.loss.backward()
        self.value_network.optimizer.step()

    def simulate(self, observation, time_transition, model: BayesianTransformer):
        """
        Simulate one timestep using the sampled model,
        retain the hidden states (h) that correspond to the actions
        taken as dictated by the sampled sequences for the next timestep.
        """
        # print('time_transition:', time_transition.shape, '\n', time_transition)
        new_pred_states, rewards, terminals = model.predict(time_transition, observation, simulation=True)

        return (
            new_pred_states,
            rewards,
            terminals
        )

    def compute_targets(self, rewards: torch.tensor, next_states: torch.tensor, terminals: torch.tensor, target_net: torch.nn.Sequential):
        """
        Learns value of states
        """
        target_net.to(device=self.device)
        rewards = rewards.reshape(-1, self.num_actions)
        terminals = terminals.reshape(-1, self.num_actions)
        if self.onehot_inputs:
            next_states = project_onehot(next_states)
        values = target_net(next_states)
        v = values.detach().flatten().reshape(-1, self.num_actions)
        target_value, _ = torch.max(
            rewards + (self.discount * v * (terminals < TP_THRESHOLD)), dim=1
        )

        return target_value.reshape(-1, 1)

    def train_(self, model: BayesianTransformer, dataset: Dataset):
        """
        Update the value network using B sequences of length L for the specified number of training iterations (kappa).
        For the states in all B sequences at a given timestep, the next states and rewards are simulated in parallel
        using the current sampled model. The simulated elements are used to compute the targets with which the value
        network is updated.
        """
        # print('\nTRAINING POLICY BTRL')
        for epoch in range(self.training_iterations):
            o, a, _, _, _, timesteps = dataset.sample_sequences()
            seq_length = len(o[0])
            # print('length:',length)

            for idx in range(seq_length):
                if idx % self.target_update_freq == 0:
                    self.target_net = deepcopy(self.value_network.layers)

                current_state = o[:, idx].unsqueeze(1)
                current_timestep = timesteps[:,idx]

                time_transitions = current_timestep

                # print('inputs:',inputs.shape)
                observation = current_state.reshape(dataset.batch_size, -1, model.state_size)
                # print('observation:',observation.shape)

                new_pred_states, rewards, terminals = self.simulate(observation, time_transitions, model)

                targets = self.compute_targets(
                    rewards,
                    new_pred_states,
                    terminals,
                    self.target_net
                )

                # Value net learns V(current state) <- max_a[r + gamma V(next)],
                # so the value-net input is the CURRENT state (last context slot),
                # not the predicted next state. At action time select_action feeds
                # predicted next states, which then correctly reads V(next).
                value_input = observation[:, -1, :]
                self.train_iter(value_input, targets)

        dataset.logger.add_scalars("Loss/Value", self.value_network.loss.item())
