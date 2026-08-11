"""
LayerNorm-MLP encoder used as the front-end of the TD3 actor and critic.

A plain MlpPolicy converges to a fuel-wasteful local optimum here: it docks by
brute-force thrusting instead of exploiting the slow orbital transfer. Putting
LayerNorm between hidden layers is the standard TD3 stabilizer -- it keeps
critic activations bounded so value targets survive the long horizon, and lets
a wider net train without the actor saturating.

DEPTH LIMIT (measured): stacking too many encoder layers re-creates the
init-saturation failure -- the actor pins at +-max_dv from step 0, every
episode flies to the boundary, and the policy gradient is flat. relu trains at
n_blocks 1-2 and fails at 3; silu trains only at 1. Hence the relu + n_blocks=2
default in constants.py. Pushing deeper also needs a smaller actor output init.
"""

from __future__ import annotations

import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


ACTIVATIONS = {
    "silu": nn.SiLU,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}
_ACTIVATIONS = ACTIVATIONS      # legacy alias


class SmartEncoder(BaseFeaturesExtractor):
    """Stack of Linear -> LayerNorm -> activation layers. The actor and critic
    each get their own instance (share_features_extractor=False), so the critic
    can learn a value-oriented representation independent of the policy."""

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        n_blocks: int = 2,
        activation: str = "relu",
    ):
        super().__init__(observation_space, features_dim)
        act = _ACTIVATIONS[activation]
        layers: list[nn.Module] = []
        prev = int(observation_space.shape[0])
        for _ in range(max(1, n_blocks)):
            layers += [nn.Linear(prev, features_dim), nn.LayerNorm(features_dim), act()]
            prev = features_dim
        self.net = nn.Sequential(*layers)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.net(observations)


def shrink_actor_output_init(model, weight_bound: float = 3e-3) -> None:
    """Re-initialize the actor's final layer small (the DDPG/TD3 fan-in trick)
    so the policy starts near zero thrust instead of saturating tanh at
    +-max_dv. Without it a wide net never learns even the easiest curriculum
    stage. Must run after construction and before learning; the actor_target is
    re-synced since SB3 copies actor -> target at build time."""
    linears = [m for m in model.actor.mu.modules() if isinstance(m, nn.Linear)]
    last = linears[-1]
    nn.init.uniform_(last.weight, -weight_bound, weight_bound)
    nn.init.uniform_(last.bias, -weight_bound, weight_bound)
    model.actor_target.load_state_dict(model.actor.state_dict())


def build_smart_policy_kwargs(
    net_arch: list[int],
    features_dim: int = 256,
    n_blocks: int = 2,
    activation: str = "relu",
) -> dict:
    """policy_kwargs swapping MlpPolicy's front-end for SmartEncoder.
    `net_arch` sizes the actor/critic heads that sit on top."""
    return dict(
        features_extractor_class=SmartEncoder,
        features_extractor_kwargs=dict(
            features_dim=features_dim,
            n_blocks=n_blocks,
            activation=activation,
        ),
        share_features_extractor=False,
        net_arch=net_arch,
        activation_fn=_ACTIVATIONS[activation],
    )
