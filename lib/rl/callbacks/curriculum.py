"""Curriculum callbacks: one authority per axis, shared across all sub-envs."""

from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class _DockRateCurriculum(BaseCallback):
    """Advances on a cleared dock-rate window, regresses after a sustained
    stall so a stage the policy cannot yet clear never freezes the run."""

    def __init__(self, n_envs, window=20, threshold=0.5, stall_patience=3):
        super().__init__(verbose=0)
        self.n_envs = n_envs
        self.threshold = threshold
        self.stall_patience = stall_patience
        self._recent = deque(maxlen=window)
        self._stalled = 0

    def _collect(self) -> bool:
        """True once a full window of episode outcomes is available."""
        for i in range(self.n_envs):
            if self.locals["dones"][i]:
                self._recent.append(bool(self.locals["infos"][i].get("docked", False)))
        return len(self._recent) >= self._recent.maxlen

    def _decide(self):
        """Returns "advance", "regress" or None, and clears the window."""
        cleared = np.mean(self._recent) >= self.threshold
        self._recent.clear()
        if cleared:
            self._stalled = 0
            return "advance"
        self._stalled += 1
        if self._stalled >= self.stall_patience:
            self._stalled = 0
            return "regress"
        return None


class DistanceCurriculum(_DockRateCurriculum):
    """Ramps the spawn distance from `min_distance` to `max_distance`."""

    def __init__(self, n_envs, increment, max_distance, min_distance, **kw):
        super().__init__(n_envs, **kw)
        self.increment = increment
        self.max_distance, self.min_distance = max_distance, min_distance
        self.distance = None
        self.locked = False

    def _on_training_start(self) -> None:
        self.distance = self.training_env.get_attr("curriculum_distance")[0]

    def lock(self, reason="another curriculum active") -> None:
        """Freezes the distance so a dip in dock rate while another axis is
        being tightened cannot regress it out from under that axis."""
        if not self.locked:
            self.locked = True
            print(f"[distance] locked at {self.distance:.1f} m ({reason})")

    @property
    def progress(self) -> str:
        return f"{sum(self._recent)}/{len(self._recent)}"

    def _push(self, distance):
        self.distance = distance
        self.training_env.env_method("set_curriculum_distance", distance)

    def _on_step(self) -> bool:
        if self.locked or not self._collect():
            return True
        action = self._decide()
        if action == "advance" and self.distance < self.max_distance:
            self._push(min(self.distance + self.increment, self.max_distance))
            print(f"[distance] advancing to {self.distance:.1f} m")
        elif action == "regress" and self.distance > self.min_distance:
            self._push(max(self.distance - self.increment, self.min_distance))
            print(f"[distance] stalled -> regressing to {self.distance:.1f} m")
        return True


class AngleCurriculum(_DockRateCurriculum):
    """Widens the "random" start-direction SECTOR, gated behind the distance
    ramp and locking it on activation.

    The sector is a RANGE, not a moving point: each episode draws uniformly
    from +-half_width around a V-bar axis, so widening strictly adds new
    directions and never stops training the ones already learned. At the 90 deg
    max it is exactly the uniform draw over the full circle.

    Sampling the whole circle from step one instead drives the policy to
    brute-force thrusting at ~11x optimal, because the optimal coasting
    trajectory has ~96% excursion margin at V-bar but only ~20% off-axis.

    Gated on the DETERMINISTIC evaluation dock rate, not the behaviour
    policy's. The docking basin is ~0.006 wide in action space, so exploration
    noise alone holds the noisy dock rate far below any sensible threshold even
    when the policy is fine: a live run sat at 11% noisy against 58% actual,
    and the sector could never widen. `eval_source` is the BestModelEval
    callback; without one this falls back to the noisy window.
    """

    def __init__(self, n_envs, start_deg, max_deg, increment_deg, distance_curriculum,
                 eval_source=None, **kw):
        super().__init__(n_envs, **kw)
        self.start_deg, self.max_deg = start_deg, max_deg
        self.increment_deg = increment_deg
        self.distance_curriculum = distance_curriculum
        self.eval_source = eval_source
        self.half_width_deg = None
        self._last_eval_step = None

    def _on_training_start(self) -> None:
        # Adopt whatever the envs were built with, so resuming a partly-widened
        # run does not throw the sector away.
        self.half_width_deg = self.training_env.get_attr("angle_half_width_deg")[0]
        if self._distance_ready():
            self._lock()

    def _distance_ready(self) -> bool:
        dc = self.distance_curriculum
        return dc is None or dc.distance is None or dc.distance >= dc.max_distance

    def _lock(self):
        if self.distance_curriculum is not None:
            self.distance_curriculum.lock("angle curriculum active")

    @property
    def progress(self) -> str:
        return f"{self.half_width_deg:.2f}deg"

    def _push(self, half_width):
        self.half_width_deg = float(np.clip(half_width, self.start_deg, self.max_deg))
        self.training_env.env_method("set_angle_half_width", self.half_width_deg)

    def _eval_decision(self):
        """Advance/regress from the newest deterministic evaluation, or None if
        no fresh one has landed since the last decision."""
        src = self.eval_source
        if src is None or src.last_eval_step is None:
            return None
        if src.last_eval_step == self._last_eval_step:
            return None
        self._last_eval_step = src.last_eval_step
        if src.last_dock_rate >= self.threshold:
            self._stalled = 0
            return "advance"
        self._stalled += 1
        if self._stalled >= self.stall_patience:
            self._stalled = 0
            return "regress"
        return None

    def _on_step(self) -> bool:
        if not self._distance_ready():
            return True
        self._lock()
        if self.eval_source is not None:
            action = self._eval_decision()
        elif self._collect():
            action = self._decide()
        else:
            return True
        if action == "advance" and self.half_width_deg < self.max_deg:
            self._push(self.half_width_deg + self.increment_deg)
            print(f"[angle] widening sector to +-{self.half_width_deg:.2f} deg")
        elif action == "regress" and self.half_width_deg > self.start_deg:
            self._push(self.half_width_deg - self.increment_deg)
            print(f"[angle] stalled -> narrowing sector to +-{self.half_width_deg:.2f} deg")
        return True
