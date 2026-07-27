"""Periodic checkpoints and best-model evaluation."""

import json
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from config import MAX_STEPS, OMEGA
from lib.rl.env import CWRendezvousEnv
from lib.rl.obs import NormalizedObsEnv
from lib.rl.symmetry import CanonicalizeDirectionEnv


def curriculum_state(distance_cb, angle_cb, rounded=True):
    """(distance, angle_half_width) with None for an inactive axis.

    Rounded because repeated widen/narrow arithmetic leaves the sector at
    0.10000000000009: the pair is used as a dict key and in an ordering
    comparison, so without rounding one stage becomes many keys and can even
    compare as "easier" than itself, locking out saves.
    """
    distance = distance_cb.distance if distance_cb is not None else None
    angle = angle_cb.half_width_deg if angle_cb is not None else None
    if rounded:
        distance = None if distance is None else round(float(distance), 3)
        angle = None if angle is None else round(float(angle), 3)
    return distance, angle


class Checkpointer(BaseCallback):
    """Saves every `freq` timesteps, keeping the last `keep`. Writes a
    .curriculum.json sidecar too: SB3 saves only the policy and optimizer, but
    curriculum state lives in the callbacks."""

    def __init__(self, save_dir: Path, prefix, n_envs, freq, keep,
                 distance_cb=None, angle_cb=None):
        super().__init__(verbose=0)
        self.save_dir, self.prefix, self.keep = save_dir, prefix, keep
        self.freq_calls = max(freq // max(n_envs, 1), 1)
        self.distance_cb, self.angle_cb = distance_cb, angle_cb

    def _on_step(self) -> bool:
        if self.n_calls % self.freq_calls:
            return True
        self.save_dir.mkdir(parents=True, exist_ok=True)
        path = self.save_dir / f"{self.prefix}_{self.num_timesteps}_steps"
        self.model.save(str(path))

        distance, angle = curriculum_state(self.distance_cb, self.angle_cb, rounded=False)
        if distance is not None:
            data = {"curriculum_distance": distance}
            if angle is not None:
                data["angle_half_width_deg"] = angle
            path.with_suffix(".curriculum.json").write_text(json.dumps(data))

        saved = sorted(self.save_dir.glob(f"{self.prefix}_*_steps.zip"),
                       key=lambda p: p.stat().st_mtime)
        for old in saved[:-self.keep]:
            old.unlink(missing_ok=True)
            old.with_suffix(".curriculum.json").unlink(missing_ok=True)
        return True


class BestModelEval(BaseCallback):
    """Evaluates the DETERMINISTIC policy on a private env and keeps the best.

    Nothing else preserves a good policy, and the training log shows the NOISY
    behaviour policy -- for a solution this narrow a very different number. A
    refinement run reached ~95% dock rate then collapsed to 0% once noise hit
    its floor, and every surviving checkpoint was worse than its own seed.

    Scores are per curriculum stage, and the file is written only at the
    hardest stage reached: a regression to an easier sector would otherwise
    overwrite a harder-stage best with a score that is higher only because the
    task got easier.
    """

    def __init__(self, scenario, save_path: Path, n_envs, freq, n_episodes,
                 distance_cb=None, angle_cb=None):
        super().__init__(verbose=0)
        self.scenario, self.save_path, self.n_episodes = scenario, save_path, n_episodes
        self.freq_calls = max(freq // max(n_envs, 1), 1)
        self.distance_cb, self.angle_cb = distance_cb, angle_cb
        self._best_by_stage = {}
        self._hardest = None
        self.best_at = None
        # Latest deterministic result, consumed by AngleCurriculum: the noisy
        # dock rate is unusable as a curriculum gate on a basin this narrow.
        self.last_dock_rate = None
        self.last_eval_step = None
        self._raw = self._env = None

    def _on_training_start(self) -> None:
        self._raw = CWRendezvousEnv(omega=OMEGA, scenario=self.scenario)
        self._env = NormalizedObsEnv(CanonicalizeDirectionEnv(self._raw))

    def _sync(self):
        distance, angle = curriculum_state(self.distance_cb, self.angle_cb)
        if distance is not None:
            self._raw.set_curriculum_distance(distance)
        if angle is not None:
            self._raw.set_angle_half_width(angle)
        return distance, angle

    @staticmethod
    def _hardness(stage):
        return tuple(-np.inf if v is None else float(v) for v in stage)

    def _evaluate(self):
        returns, docks = [], 0
        for _ in range(self.n_episodes):
            obs, _ = self._env.reset()
            total = 0.0
            for _ in range(MAX_STEPS):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self._env.step(action)
                total += reward
                if terminated or truncated:
                    break
            returns.append(total)
            docks += bool(info.get("docked", False))
        return float(np.mean(returns)), docks / self.n_episodes

    def _on_step(self) -> bool:
        if self.n_calls % self.freq_calls:
            return True
        stage = self._sync()
        hardness = self._hardness(stage)
        if self._hardest is None or hardness > self._hardest:
            self._hardest = hardness

        score, dock_rate = self._evaluate()
        self.last_dock_rate, self.last_eval_step = dock_rate, self.num_timesteps
        prev = self._best_by_stage.get(stage, -np.inf)
        self._best_by_stage[stage] = max(prev, score)

        label = f"d={stage[0]:.0f}m" if stage[0] is not None else "d=-"
        if stage[1] is not None:
            label += f" a=+-{stage[1]:.2f}deg"
        head = (f"[eval] {self.num_timesteps} [{label}]: return {score:.1f}, "
                f"dock {100 * dock_rate:.0f}%")

        if hardness < self._hardest:
            print(f"{head} (curriculum regressed - best not touched)")
        elif score > prev:
            self.best_at = self.num_timesteps
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(self.save_path))
            sidecar = {"curriculum_distance": stage[0], "angle_half_width_deg": stage[1],
                       "num_timesteps": int(self.num_timesteps),
                       "eval_mean_return": score, "eval_dock_rate": dock_rate}
            self.save_path.with_suffix(".curriculum.json").write_text(
                json.dumps({k: v for k, v in sidecar.items() if v is not None}))
            print(f"{head} -> NEW BEST")
        else:
            print(f"{head} (best here {prev:.1f} @ {self.best_at})")
        return True
