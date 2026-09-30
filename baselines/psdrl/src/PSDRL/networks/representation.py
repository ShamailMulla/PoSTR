import torch
import torch.nn as nn

from ..common.settings import CONV_FEAT, KERNEL, STRIDE, AE_LOSS_F, AE_OPTIM
from ..common.utils import expand_image


class IdentityEncoder(nn.Module):
    """Passthrough encoder for vector observations (no CNN, no compression).

    Used when embed_dim == obs_dim (e.g. DeepSea 5x5: 25-dim one-hot).
    Satisfies the same interface as AutoEncoder so the rest of PSDRL is unchanged.
    """

    def __init__(self, config: dict, device: str):
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config["learning_rate"])
        self.loss = torch.tensor(0.0)
        self.to(device)

    def embed(self, obs: torch.Tensor) -> torch.Tensor:
        # Single obs (D,) → (1, D) to match CNN encoder's batched output shape.
        # Batched obs (B, D) → (B, D) unchanged.
        return obs.float().unsqueeze(0) if obs.dim() == 1 else obs.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.float()


class AutoEncoder(nn.Module):
    def __init__(self, config: dict, device: str):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.optimizer = AE_OPTIM(self.parameters(), lr=config["learning_rate"])
        self.loss_function = AE_LOSS_F
        self.loss = 0
        self.to(device)

    def embed(self, obs: torch.tensor):
        with torch.no_grad():
            return self.encoder.forward(obs)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, CONV_FEAT, KERNEL, stride=STRIDE),
            nn.ReLU(),
            nn.Conv2d(CONV_FEAT, CONV_FEAT * 2, KERNEL, stride=STRIDE),
            nn.ReLU(),
            nn.Conv2d(CONV_FEAT * 2, CONV_FEAT * 4, KERNEL, stride=STRIDE),
            nn.ReLU(),
            nn.Conv2d(CONV_FEAT * 4, CONV_FEAT * 8, KERNEL, stride=STRIDE),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, obs: torch.tensor):
        if len(obs.shape) < 4:
            obs = expand_image(obs)
        x = obs.transpose(0, 1)
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(CONV_FEAT * 8, 2, 2))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(CONV_FEAT * 8, CONV_FEAT * 4, KERNEL, stride=STRIDE),
            nn.ReLU(),
            nn.ConvTranspose2d(CONV_FEAT * 4, CONV_FEAT * 2, KERNEL, stride=STRIDE),
            nn.ReLU(),
            nn.ConvTranspose2d(CONV_FEAT * 2, CONV_FEAT, KERNEL, stride=STRIDE),
            nn.ReLU(),
            nn.ConvTranspose2d(
                CONV_FEAT, 1, KERNEL + 1, stride=STRIDE, output_padding=1
            ),
        )

    def forward(self, embed: torch.tensor):
        x = self.unflatten(embed)
        x = self.decoder_conv(x).squeeze()
        return x
