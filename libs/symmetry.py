"""
Exploits the exact mirror symmetry of the CW rendezvous MDP.

CW dynamics are linear and every reward term (reward_pos, reward_terminal)
is built from norms / abs() values, so the whole MDP is invariant under the
joint transform:
    (position, velocity, action) -> -(position, velocity, action)
(dv_used is a scalar magnitude and is left untouched by the mirror.)

Rather than hope a generic MLP discovers this symmetry from data alone —
unreliable, since a policy trained mostly on x>0 has to *extrapolate* to
x<0, which is exactly what causes a trained policy to apply the wrong-side
orbit — this wrapper enforces it structurally: whenever x<0 the observation
handed to the policy is mirrored into the x>=0 canonical view, and the
resulting action is mirrored back before being applied to the real
environment. The policy only ever has to solve the x>=0 half of the
problem; the x<0 half is exact by construction, not learned.

Must wrap the RAW env (before NormalizedObsEnv) — it needs the true signed
physical x, not a normalized-to-[0,1] value.
"""

import numpy as np
import gymnasium as gym


class CanonicalizeDirectionEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._mirror = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        canon_obs, self._mirror = self._canonicalize(obs)
        return canon_obs, info

    def step(self, action):
        real_action = -np.asarray(action) if self._mirror else action
        obs, reward, terminated, truncated, info = self.env.step(real_action)
        canon_obs, self._mirror = self._canonicalize(obs)
        return canon_obs, reward, terminated, truncated, info

    @staticmethod
    def _canonicalize(obs: np.ndarray):
        mirror = obs[0] < 0
        if not mirror:
            return obs, False
        canon = obs.copy()
        canon[:-1] *= -1  # negate everything except the trailing dv_used scalar
        return canon, True
