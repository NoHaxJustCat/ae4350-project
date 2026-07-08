import torch
import torch.nn as nn

from libs.constants import OBS_DIM, ACTION_DIM


class Critic(nn.Module):
    """
    Q-value network Q(s, a) → scalar.

    State  dim : OBS_DIM    — 5 (2-D mode) or 7 (3-D mode)
    Action dim : ACTION_DIM — 2 (2-D mode) or 3 (3-D mode)

    State and action are embedded separately before being concatenated,
    which gives the network more flexibility to learn the interaction between
    position/velocity and the commanded burn.
    """

    def __init__(
        self,
        state_dim:  int = OBS_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.state_net  = nn.Linear(state_dim,  hidden_dim)
        self.action_net = nn.Linear(action_dim, hidden_dim)

        self.combined_net = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        unbatched = state.dim() == 1
        if unbatched:
            state  = state.unsqueeze(0)
            action = action.unsqueeze(0)

        s_out = torch.relu(self.state_net(state))
        a_out = torch.relu(self.action_net(action))
        x     = torch.cat([s_out, a_out], dim=1)
        out   = self.combined_net(x)
        return out.squeeze(0) if unbatched else out