"""Logging, throughput, noise decay and live diagnostics."""

import json
import math
import time
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from config import (
    ACTION_IMPULSE_DIM, DOCK_RATE_WINDOW, ENV_MAX_DV_COEFF, LOG_EVERY,
    NOISE_DECAY_FRAC, OU_STD_PER_SIGMA, SMOOTHING_WINDOW,
)
from lib.plots.diagnostics import write_panels
from lib.plots.trajectory import plot_trajectory

HISTORY_KEYS = ("rewards", "steps", "delta_v", "dv_ratio", "docked", "r_pos",
                "r_fuel", "r_term", "noise_std", "curriculum_distance",
                "angle_half_width_deg", "actor_loss", "critic_loss")


def json_number(value):
    """None or any non-finite float -> None (JSON null). json.dumps emits BARE
    NaN tokens, which are invalid JSON; the dashboard parses status.json inside
    a bare `catch {}`, so one NaN made a whole run vanish from it."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


class NoiseDecay(BaseCallback):
    """Anneals exploration noise START -> END over the first `decay_frac` of the
    run. Constant noise forces nonzero actions forever, starving the agent of
    ever seeing a "coast, don't burn" episode."""

    def __init__(self, total_timesteps, sigma_start, sigma_end,
                 decay_frac=NOISE_DECAY_FRAC, start_timesteps=0):
        super().__init__(verbose=0)
        self.sigma_start, self.sigma_end = sigma_start, sigma_end
        self.start_timesteps = start_timesteps
        self.decay_steps = max(1, int(total_timesteps * decay_frac))

    def _on_step(self) -> bool:
        noise = self.model.action_noise
        if noise is None:
            return True
        done = max(0, self.num_timesteps - self.start_timesteps)
        progress = min(1.0, done / self.decay_steps)
        sigma = self.sigma_start + progress * (self.sigma_end - self.sigma_start)
        for sub in getattr(noise, "noises", [noise]):
            sub._sigma[:] = sigma
        return True


class Throughput(BaseCallback):
    """Steps/sec on a fixed timestep cadence; episode length varies too much
    for a per-episode timer to be a clean signal."""

    def __init__(self, every=2000):
        super().__init__(verbose=0)
        self.every = every
        self._t = self._steps = None

    def _on_training_start(self) -> None:
        self._t, self._steps = time.perf_counter(), self.num_timesteps

    def _on_step(self) -> bool:
        if self.num_timesteps - self._steps < self.every:
            return True
        now = time.perf_counter()
        dt, dn = now - self._t, self.num_timesteps - self._steps
        if dt > 0:
            print(f"[throughput] {self.num_timesteps:>10} | {dn / dt:>8.1f} steps/s | "
                  f"{1000 * dt / dn:.2f} ms/step")
        self._t, self._steps = now, self.num_timesteps
        return True


class EpisodeLogger(BaseCallback):
    """Accumulates per-episode diagnostics across N parallel envs.

    Trajectories read info["state"] / info["applied_action"] (raw physical
    values), not new_obs, which is normalized policy input and would corrupt
    the plots.
    """

    def __init__(self, episodes_dir: Path, n_envs, log_every=LOG_EVERY,
                 distance_cb=None, angle_cb=None):
        super().__init__(verbose=0)
        self.episodes_dir, self.n_envs, self.log_every = episodes_dir, n_envs, log_every
        self.distance_cb, self.angle_cb = distance_cb, angle_cb
        self.history = {k: [] for k in HISTORY_KEYS}
        self._acc = [self._new_acc() for _ in range(n_envs)]
        self._episode = 0
        print(f"{'ep':>6} | {'steps':>5} | {'reward':>9} | {'r_pos':>8} | {'r_fuel':>8} | "
              f"{'r_term':>8} | {'noise':>7} | {'dv/opt':>7} | {'dist':>6} | "
              f"{'docked':>6} | {'prog':>8}")

    @staticmethod
    def _new_acc():
        return {"states": [], "actions": [], "delta_v": 0.0, "r_pos": 0.0,
                "r_fuel": 0.0, "r_term": 0.0, "docked": False}

    def _noise_std(self) -> float:
        noise = self.model.action_noise
        if noise is None:
            return 0.0
        sub = getattr(noise, "noises", [noise])[0]
        return float(np.mean(sub._sigma)) * OU_STD_PER_SIGMA

    def _seed_start(self, i):
        infos = getattr(self.training_env, "reset_infos", None)
        state = infos[i].get("state") if infos and i < len(infos) else None
        if state is not None:
            self._acc[i]["states"].append(np.asarray(state).copy())
            self._acc[i]["actions"].append(np.zeros(ACTION_IMPULSE_DIM))

    def _on_training_start(self) -> None:
        for i in range(self.n_envs):
            self._seed_start(i)

    def _losses(self):
        """SB3 stashes the latest train/* values on its own logger; they are
        NaN until the replay buffer passes learning_starts."""
        values = getattr(self.model.logger, "name_to_value", {})
        return (float(values.get("train/actor_loss", float("nan"))),
                float(values.get("train/critic_loss", float("nan"))))

    def _record(self, info, acc):
        dv_opt = info.get("dv_opt", float("nan"))
        ep = info.get("episode", {})
        actor_loss, critic_loss = self._losses()
        return {
            "rewards": ep.get("r", 0.0), "steps": ep.get("l", 1),
            "delta_v": acc["delta_v"],
            "dv_ratio": acc["delta_v"] / dv_opt if dv_opt else float("nan"),
            "docked": acc["docked"], "r_pos": acc["r_pos"], "r_fuel": acc["r_fuel"],
            "r_term": acc["r_term"], "noise_std": self._noise_std(),
            "curriculum_distance": info.get("curriculum_distance", float("nan")),
            "angle_half_width_deg": (info.get("angle_half_width_deg", float("nan"))
                                     if self.angle_cb else float("nan")),
            "actor_loss": actor_loss, "critic_loss": critic_loss,
        }

    def _on_step(self) -> bool:
        for i in range(self.n_envs):
            info, acc = self.locals["infos"][i], self._acc[i]
            substates = info.get("substates") or [info["state"]]
            for s in substates:
                acc["states"].append(np.asarray(s).copy())
                acc["actions"].append(np.zeros(ACTION_IMPULSE_DIM))
            applied = np.asarray(info.get("applied_action", np.zeros(ACTION_IMPULSE_DIM)))
            if len(acc["actions"]) > len(substates):
                acc["actions"][-len(substates) - 1] = applied.copy()
            acc["delta_v"] += info.get("delta_v", 0.0)
            acc["r_pos"] += info.get("reward_pos", 0.0)
            acc["r_fuel"] += info.get("reward_fuel", 0.0)
            acc["r_term"] += info.get("reward_terminal", 0.0) + info.get("reward_stop", 0.0)
            acc["docked"] |= bool(info.get("docked", False))

            if not self.locals["dones"][i]:
                continue

            rec = self._record(info, acc)
            for k, v in rec.items():
                self.history[k].append(v)

            if self._episode % self.log_every == 0:
                dock = 100 * float(np.mean(self.history["docked"][-DOCK_RATE_WINDOW:]))
                prog = self.distance_cb.progress if self.distance_cb else "-"
                if self.angle_cb is not None:
                    prog = self.angle_cb.progress
                print(f"{self._episode:>6} | {rec['steps']:>5} | {rec['rewards']:>9.2f} | "
                      f"{rec['r_pos']:>8.2f} | {rec['r_fuel']:>8.2f} | {rec['r_term']:>8.2f} | "
                      f"{rec['noise_std']:>7.4f} | {rec['dv_ratio']:>7.2f} | "
                      f"{rec['curriculum_distance']:>6.1f} | {dock:>5.1f}% | {prog:>8}")
                if i == 0:
                    plot_trajectory(acc["states"], acc["actions"],
                                    self.episodes_dir / f"ep_{self._episode:06d}.png",
                                    min_dv_display=ENV_MAX_DV_COEFF * 0.01)

            self._acc[i] = self._new_acc()
            self._seed_start(i)
            self._episode += 1
        return True


class LiveDiagnostics(BaseCallback):
    """Rebuilds the diagnostics figure and status.json periodically so a run
    can be watched while training. Time-throttled: rebuilding every episode
    measurably slows training."""

    def __init__(self, episode_logger, paths, scenario, total_timesteps,
                 every_seconds=30.0, angle_cb=None):
        super().__init__(verbose=0)
        self.episodes, self.paths = episode_logger, paths
        self.scenario, self.total_timesteps = scenario, total_timesteps
        self.every_seconds = every_seconds
        self.angle_cb = angle_cb
        self._last = self._start = None

    def _on_training_start(self) -> None:
        self._last = self._start = time.perf_counter()

    def write_status(self, now: float) -> None:
        h = self.episodes.history
        elapsed = now - self._start
        angle = self.angle_cb.half_width_deg if self.angle_cb is not None else None
        status = {
            "scenario": self.scenario, "run_tag": self.paths.run_tag,
            "num_timesteps": int(self.num_timesteps),
            "total_timesteps": int(self.total_timesteps),
            "episode_count": len(h["rewards"]),
            "recent_dock_rate": json_number(np.mean(h["docked"][-DOCK_RATE_WINDOW:])
                                            if h["docked"] else None),
            "recent_avg_reward": json_number(np.mean(h["rewards"][-SMOOTHING_WINDOW:])
                                             if h["rewards"] else None),
            "curriculum_distance": json_number(h["curriculum_distance"][-1]
                                               if h["curriculum_distance"] else None),
            "angle_half_width_deg": json_number(angle),
            "elapsed_seconds": json_number(elapsed),
            "steps_per_sec": json_number(self.num_timesteps / elapsed if elapsed > 0 else None),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        # allow_nan=False so a future field that forgets json_number fails here
        # rather than silently dropping the run; caught so a diagnostics write
        # can never kill a multi-hour training run.
        try:
            payload = json.dumps(status, allow_nan=False)
        except ValueError as exc:
            print(f"WARNING: status.json not written - non-finite value: {exc}")
            return
        self.paths.status.write_text(payload)

    def _on_step(self) -> bool:
        now = time.perf_counter()
        if now - self._last < self.every_seconds:
            return True
        self._last = now
        if self.episodes.history["rewards"]:
            write_panels(self.episodes.history, self.scenario, self.paths.panels)
        self.write_status(now)
        return True
