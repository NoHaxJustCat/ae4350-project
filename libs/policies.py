"""
Custom "smart" network architecture for the CW rendezvous TD3 agent.

Motivation
----------
The plain [128,128] / [128,128,128] MlpPolicy runs converge to a
fuel-wasteful local optimum: they dock in ~28 steps by burning ~1.5 m/s
(~130x the optimal V-bar two-impulse Δv) instead of exploiting the slow,
near-zero-thrust orbital transfer that the classical analytic solution uses.
Two things a plain MLP struggles with on THIS problem:

  1. Credit assignment over ~1000+ steps (γ=0.9995). A vanilla TD3 critic
     is prone to value over-estimation / divergence at high γ, so the tiny
     terminal fuel bonus that rewards a long, patient coast never propagates
     cleanly back to the early "coast, don't burn" decisions.

  2. Representing a policy that outputs *near-exactly zero* thrust for
     hundreds of consecutive steps and then a precise small correction — a
     delicate function a shallow MLP smears out.

Design
------
The proven, low-risk way to give a TD3 agent both more capacity AND a smoother
value landscape is **LayerNorm between every hidden layer** — the single most
reliable DDPG/TD3 stabilizer for high-γ continuous control (cf. the original
LayerNorm-in-RL results, D4PG, CrossQ, TD7). It keeps critic activations
bounded so value targets don't blow up over the long horizon, and it lets a
deeper/wider net train without the actor saturating.

So the "smart" encoder here is simply a deeper, wider MLP with
Linear -> LayerNorm -> SiLU at each layer, exposed to SB3 as a
BaseFeaturesExtractor. The actor/critic heads on top are the usual SB3 MLP
heads (their width is the `net_arch` passed from training.py), and each gets
its OWN encoder instance (share_features_extractor=False in training.py) so
the critic can learn a value-oriented representation independent of the policy.

An earlier revision of this file used residual blocks + aggressive orthogonal
init; on this task that combination saturated the actor (it drove straight to
the boundary every episode and never learned even the trivial 10 m curriculum
stage). Plain LayerNorm-MLP with torch-default init trains cleanly, so that is
what this is now. Weights are left at torch defaults on purpose — SB3's TD3
relies on them and they are well-behaved with LayerNorm.

DEPTH LIMIT (measured, not guessed). Stacking too many encoder layers on top
of the [256,256] actor/critic heads re-creates the same init-saturation: the
compounded pre-tanh magnitude pins the actor at ±max_dv from step 0, so every
episode flies straight to the boundary, the replay buffer fills with uniformly
bad transitions, and the deterministic policy gradient is flat — no learning,
ever. An A/B isolation sweep on the 10 m curriculum stage found the trainable
frontier: relu trains at n_blocks ∈ {1,2} and fails at 3; silu trains only at
n_blocks=1 and fails at 2 (silu's smooth, unbounded positive tail saturates
more easily). So the default is relu + n_blocks=2 — the deepest LayerNorm
encoder that reliably learns. If you want to push depth further, you must also
shrink the actor's final-layer init so initial actions start near zero.
"""

from __future__ import annotations

import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


_ACTIVATIONS = {
    "silu": nn.SiLU,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}


class SmartEncoder(BaseFeaturesExtractor):
    """Deeper/wider LayerNorm-MLP state encoder used as the front-end of both
    the actor and the critic (each gets its own instance when
    share_features_extractor=False).

    Each hidden layer is Linear -> LayerNorm -> activation. The output
    embedding is itself LayerNorm'd (via the last layer) so the critic's
    obs-branch input is well-conditioned before it is concatenated with the
    action.

    Parameters
    ----------
    features_dim : width of every hidden layer and of the output embedding.
    n_blocks     : number of Linear->LayerNorm->activation layers.
    activation   : one of _ACTIVATIONS keys.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        n_blocks: int = 2,
        activation: str = "relu",
    ):
        super().__init__(observation_space, features_dim)
        act = _ACTIVATIONS[activation]
        in_dim = int(observation_space.shape[0])

        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(max(1, n_blocks)):
            layers += [nn.Linear(prev, features_dim), nn.LayerNorm(features_dim), act()]
            prev = features_dim
        self.net = nn.Sequential(*layers)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.net(observations)


def shrink_actor_output_init(model, weight_bound: float = 3e-3) -> None:
    """Re-initialize the actor's FINAL linear layer with small uniform weights
    (the classic DDPG/TD3 fan-in trick, [-3e-3, 3e-3]) so the policy outputs
    near-zero actions at init instead of saturating tanh at ±max_dv.

    Without this, a deep or very WIDE net produces large pre-tanh values from
    step one, so the actor is pinned at the action bound, every episode flies
    straight to the boundary, the replay buffer fills with uniformly bad
    transitions, and the deterministic policy gradient is flat — it never
    learns even the trivial 10 m curriculum stage. Shrinking only the last
    layer tames the output magnitude regardless of how large the hidden
    activations are, which is what lets width/depth beyond the earlier
    saturation ceiling (see SmartEncoder docstring) actually train.

    Must run AFTER TD3 construction and BEFORE learning; the actor_target is
    re-synced to the actor so both start identical (SB3 copies actor->target
    at build time, i.e. before this edit)."""
    import torch.nn as nn
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
    """policy_kwargs for TD3 that swaps the flat MlpPolicy front-end for the
    LayerNorm-MLP SmartEncoder. `net_arch` still controls the width of the
    actor/critic heads that sit on top of the encoder."""
    act = _ACTIVATIONS[activation]
    return dict(
        features_extractor_class=SmartEncoder,
        features_extractor_kwargs=dict(
            features_dim=features_dim,
            n_blocks=n_blocks,
            activation=activation,
        ),
        share_features_extractor=False,
        net_arch=net_arch,
        activation_fn=act,
    )
