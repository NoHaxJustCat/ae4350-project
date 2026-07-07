import torch
import torch.nn as nn
class Critic(nn.Module):
    def __init__(self, state_dim: int = 7, action_dim: int = 3, hidden_dim: int = 128):
        super().__init__()
        
        # Separate layers for State and Action
        self.state_net = nn.Linear(state_dim, hidden_dim)
        self.action_net = nn.Linear(action_dim, hidden_dim)
        
        # Combined layers
        self.combined_net = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # Add batch dim if needed
        unbatched = state.dim() == 1
        if unbatched:
            state = state.unsqueeze(0)
            action = action.unsqueeze(0)
        s_out = torch.relu(self.state_net(state))
        a_out = torch.relu(self.action_net(action))
        x = torch.cat([s_out, a_out], dim=1)
        out = self.combined_net(x)
        return out.squeeze(0) if unbatched else out