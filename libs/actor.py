import torch
import torch.nn as nn

from libs.constants import OBS_DIM, ACTION_DIM


class Actor(nn.Module):
    """
    Policy network π(s) → a.

    Input  dim : OBS_DIM    — 5 (2-D mode) or 7 (3-D mode)
    Output dim : ACTION_DIM — 2 (2-D mode) or 3 (3-D mode)

    Output is squashed through Tanh and scaled to [-max_action, max_action].
    """

    def __init__(
        self,
        state_dim:  int   = OBS_DIM,
        action_dim: int   = ACTION_DIM,
        max_action: float = 1.0,
        hidden_dim: int   = 128,
    ):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)