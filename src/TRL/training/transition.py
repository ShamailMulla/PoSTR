"""
Training the transformer transition network
"""
import torch
import torch.nn.functional as F

from ..common.replay import Dataset
from ..common.utils import compute_state_loss


class TransitionModelTrainer:
    def __init__(
        self,
        config: dict,
        transition_network: torch.nn.Module,
        terminal_network: torch.nn.Module,
        batch_size: int,
        num_actions: int,
        device: str,
    ):
        self.device = device
        self.num_actions = num_actions
        self.context_length = config['context_length']

        self.window_length = config["window_length"]
        self.training_iterations = config["training_iterations"]

        self.obs_type  = config.get("obs_type", "grid")
        self.obs_space = transition_network.states

        self.terminal_network = terminal_network
        self.transition_network = transition_network
        self.networks = [self.transition_network, self.terminal_network]


    def train_(self, dataset: Dataset):
        """
        Update the recurrent transition model and the terminal model simultaneously using B sequences of length L for
        the specified number of training iterations. Gradients are accumulated for the specified window length, after
        which they are back-propagated.
        """
        # print('TRAINING TRANSITION NETWORK')
        for _ in range(self.training_iterations):
            o, a, o1, r, done, timesteps = dataset.sample_sequences()
            ##### stack previous and current state@!!!!
            seq_length = len(o[0])
            # print('o.shape:',len(o),seq_length)
            # print('Sampled timesteps:',timesteps.shape)

            for net in self.networks:
                net.loss = 0
            window_idx = 0
            for idx in range(seq_length):
                # print('\nidx:',idx)
                for net in self.networks:
                    net.optimizer.zero_grad()

                current_state = o[:, idx].unsqueeze(1)
                next_state = o1[:, idx]
                current_timestep = timesteps[:,idx]
                time_transitions = current_timestep
                observation = current_state.reshape(dataset.batch_size, -1, self.transition_network.states)

                # print('time_transitions:',time_transitions.shape,'\n',time_transitions)

                # print('current_state:',current_state.shape)
                # print('next_state:',next_state.shape)
                # print('current_state:',observation.shape)
                next_state_preds, return_preds, _ = self.transition_network.forward(
                    time_transitions, observation, a[:, idx]
                )
                done_pred = self.terminal_network.forward(next_state)

                transition_s1_loss = compute_state_loss(
                    next_state_preds, next_state,
                    obs_type=self.obs_type,
                    obs_space=self.obs_space,
                )
                self.transition_network.loss += transition_s1_loss
                transition_r_loss = F.mse_loss(return_preds, r[:, idx].unsqueeze(1))
                self.transition_network.loss += transition_r_loss

                print('transition loss:',transition_s1_loss.item(), '\trewards loss:',transition_r_loss.item())

                self.terminal_network.loss += self.terminal_network.loss_function(
                    done[:, idx], done_pred
                )

                if window_idx == self.window_length or idx == seq_length - 1:
                    for net in self.networks:
                        net.loss /= window_idx + 1
                        net.loss.backward()
                        torch.nn.utils.clip_grad_norm_(net.parameters(), 5e-4) # clip extreme losses - more robust to outliers
                        net.optimizer.step()

                    dataset.logger.add_scalars(
                        ["Loss/Transition_s1", "Loss/Transition_r", "Loss/Terminal"],
                        [
                            transition_s1_loss.item(), transition_r_loss.item(),
                            self.terminal_network.loss.item(),
                        ],
                    )

                    self.transition_network.loss = 0
                    self.terminal_network.loss = 0
                    window_idx = 0
                else:
                    window_idx += 1

