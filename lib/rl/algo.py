"""Off-policy algorithm construction, shared by training and the ablation study.

TD3 is the nominal. SAC and DDPG exist so the report can attribute results to
the algorithm rather than to the tuning around it, so they are built with the
SAME encoder, head widths, discount, optimizer and buffer as TD3. The only
differences are the ones intrinsic to each algorithm:

    DDPG  single critic, no target-policy smoothing, no delayed policy update.
    SAC   a squashed-Gaussian actor whose entropy term IS the exploration, so
          it gets no OU noise and no shrunk output init (its log_std head, not
          the mean layer, sets the initial action spread).
"""

from stable_baselines3 import DDPG, SAC, TD3

from lib.rl.net import ACTIVATIONS, build_smart_policy_kwargs

ALGOS = {"td3": TD3, "sac": SAC, "ddpg": DDPG}

# SAC explores by sampling its own policy; handing it an external noise process
# on top would confound the comparison.
USES_ACTION_NOISE = {"td3": True, "ddpg": True, "sac": False}


def build_policy_kwargs(net_arch, features_dim, n_blocks, activation, smart=True):
    """SmartEncoder front-end, or a stock MlpPolicy for the ablation."""
    if smart:
        return build_smart_policy_kwargs(net_arch, features_dim, n_blocks, activation)
    return dict(net_arch=net_arch, activation_fn=ACTIVATIONS[activation])


def build_algo(algo, *, policy_kwargs, action_noise, td3_kwargs, **common):
    """`common` carries everything the three algorithms share verbatim."""
    cls = ALGOS[algo]
    kwargs = dict(common, policy_kwargs=policy_kwargs)
    if USES_ACTION_NOISE[algo]:
        kwargs["action_noise"] = action_noise
    if algo == "td3":
        kwargs.update(td3_kwargs)
    return cls(policy="MlpPolicy", **kwargs)
