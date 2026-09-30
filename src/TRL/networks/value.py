import torch
import torch.nn as nn

from ..common.settings import VN_LOSS_F, VN_OPTIM


class Network(nn.Module):
    def __init__(self, input_dim: int, config: dict, seq_length, device: str):
        super().__init__()
        self.input_dim = input_dim
        # hidden_layers counts the layers between the input layer and the
        # prediction head; defaults to 2, the architecture used before this
        # parameter was configurable.
        hidden_layers = int(config.get("hidden_layers", 2))
        layers = [
            nn.Flatten(), # does not count as a layer
            nn.Linear(input_dim, config["hidden_dim"]), # input layer
            nn.Tanh(),
        ]
        for _ in range(hidden_layers):
            layers += [
                nn.Linear(config["hidden_dim"], config["hidden_dim"]),
                nn.Tanh(),
            ]
        layers.append(nn.Linear(config["hidden_dim"], 1)) # pred layer
        self.layers = nn.Sequential(*layers)
        self.loss_function = VN_LOSS_F
        self.optimizer = VN_OPTIM(self.parameters(), lr=config["learning_rate"])
        self.to(device)
        self.loss = 0

    def forward(self, x: torch.tensor):
        return self.layers(x)

    def predict(self, x: torch.tensor):
        with torch.no_grad():
            return self.layers(x)
