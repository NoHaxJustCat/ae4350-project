import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, state_dim: int = 6, action_dim: int = 3, max_action: float = 1.0, hidden_dim: int = 128):
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