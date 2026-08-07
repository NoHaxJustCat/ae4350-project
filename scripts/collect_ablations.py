"""
Gather the ablation study into one self-contained directory for plotting.

    python scripts/collect_ablations.py

Reads tmp/abl_*/model/{history.npz,params.json} plus the reused nominal, and
writes for each run:

    tmp/ablations/<name>.npz      per-episode history + a "params_json" entry

so a plotting script needs exactly one file per run and never has to consult
config.py, which will have moved on by the time the report is written. A
human-readable tmp/ablations/manifest.json indexes them.

The nominal is tmp/vbar_fresh, trained before the ablation flags existed, so its
params.json is reconstructed here from the values recorded in its saved model.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import OUT_DIR

OUT = Path(OUT_DIR) / "ablations"
NOMINAL_TAG = "vbar_fresh"

# What each run is meant to show, carried into the manifest so the plots can be
# labelled without re-deriving it from the parameter diff.
LABELS = {
    "nominal":       ("Nominal", "TD3, gamma 0.99, lr 1e-4, net [256,256], replay 300k"),
    "gamma_low":     ("gamma 0.90", "Discount lowered from 0.99"),
    "lr_low":        ("lr 1e-5", "Learning rate lowered 10x"),
    "lr_high":       ("lr 1e-3", "Learning rate raised 10x"),
    "net_32":        ("net 32", "Hidden width 32 (nominal 256)"),
    "net_64":        ("net 64", "Hidden width 64 (nominal 256)"),
    "net_128":       ("net 128", "Hidden width 128 (nominal 256)"),
    "no_replay":     ("no replay", "Replay capacity cut to one batch (1280)"),
    "no_curriculum": ("no curriculum", "Spawn at 1000 m from step 0, no distance ramp"),
    "no_encoder":    ("no encoder", "Stock MlpPolicy instead of the LayerNorm encoder"),
    "sac":           ("SAC", "SB3 SAC, same net and hyperparameters"),
    "ddpg":          ("DDPG", "SB3 DDPG, same net and hyperparameters"),
}
ORDER = list(LABELS)


def nominal_params(run_dir: Path) -> dict:
    """Rebuild the nominal's params.json from its saved model and status.json.

    Read from the artefacts rather than hardcoded so this cannot silently
    disagree with what actually ran.
    """
    from stable_baselines3 import TD3

    from config import (
        ACTIVATION, ENV_CURRICULUM_BOUNDARY_MULT, ENV_CURRICULUM_INCREMENT,
        ENV_CURRICULUM_MAX_DISTANCE, ENV_CURRICULUM_START_DISTANCE, ENV_DT_AGENT,
        ENV_MAX_DV_COEFF, ENV_POS_TOLERANCE, ENV_TIMEOUT, MIN_BUFFER, N_BLOCKS,
        NUM_ENVS, OU_DT, OU_THETA, TD3_TARGET_NOISE_CLIP, TD3_TARGET_POLICY_NOISE,
        TRAIN_FREQ, GRADIENT_STEPS,
    )

    model = TD3.load(run_dir / "model" / "best_model.zip", device="cpu")
    status = json.loads((run_dir / "model" / "status.json").read_text())
    width = model.policy_kwargs["features_extractor_kwargs"]["features_dim"]
    return {
        "run_tag": NOMINAL_TAG, "scenario": status["scenario"], "algo": "td3",
        "total_timesteps": status["total_timesteps"], "seed": None,
        "gamma": model.gamma, "learning_rate": float(model.learning_rate),
        "buffer_size": model.buffer_size, "batch_size": model.batch_size,
        "tau": model.tau, "learning_starts": MIN_BUFFER, "train_freq": TRAIN_FREQ,
        "gradient_steps": GRADIENT_STEPS,
        "target_policy_noise": TD3_TARGET_POLICY_NOISE,
        "target_noise_clip": TD3_TARGET_NOISE_CLIP,
        "net_arch": list(model.policy_kwargs["net_arch"]), "features_dim": width,
        "n_blocks": N_BLOCKS, "activation": ACTIVATION, "smart_encoder": True,
        # runs/vbar.ps1 defaults, which is how this run was launched.
        "action_noise": "ou", "noise_std_start": 0.10, "noise_std_end": 0.01,
        "noise_decay_frac": 0.35, "ou_theta": OU_THETA, "ou_dt": OU_DT,
        "curriculum": True,
        "curriculum_start_distance": ENV_CURRICULUM_START_DISTANCE,
        "curriculum_max_distance": ENV_CURRICULUM_MAX_DISTANCE,
        "curriculum_increment": ENV_CURRICULUM_INCREMENT,
        "angle_curriculum": False,
        "env_dt_agent": ENV_DT_AGENT, "env_timeout": ENV_TIMEOUT,
        "pos_tolerance": ENV_POS_TOLERANCE, "max_dv_coeff": ENV_MAX_DV_COEFF,
        "boundary_mult": ENV_CURRICULUM_BOUNDARY_MULT,
        "n_envs": NUM_ENVS, "device": "cuda",
        "policy_class": "TD3Policy",
        "n_parameters": sum(p.numel() for p in model.policy.parameters()),
        "reconstructed": "Trained before the ablation flags existed; these values "
                         "are read back from the saved model, not from a live run.",
    }


def run_dir_for(name: str) -> Path:
    root = Path(OUT_DIR)
    return root / (NOMINAL_TAG if name == "nominal" else f"abl_{name}")


def summarize(history: dict) -> dict:
    """Tail statistics, for the manifest table."""
    docked = np.asarray(history["docked"], dtype=float)
    ratio = np.asarray(history["dv_ratio"], dtype=float)
    if not docked.size:
        return {}
    tail = slice(-max(1, len(docked) // 10), None)      # last 10% of episodes
    finite = ratio[tail][np.isfinite(ratio[tail])]
    rewards = np.asarray(history["rewards"], dtype=float)
    return {
        "episodes": int(docked.size),
        "dock_rate_all": round(100 * float(docked.mean()), 1),
        "dock_rate_final": round(100 * float(docked[tail].mean()), 1),
        "reward_final": round(float(rewards[tail].mean()), 1),
        "dv_ratio_final_median": (round(float(np.median(finite)), 2)
                                  if finite.size else None),
    }


def collect(name: str) -> dict | None:
    run = run_dir_for(name)
    history_path = run / "model" / "history.npz"
    if not history_path.exists():
        print(f"  {name:<14} MISSING ({history_path})")
        return None

    params_path = run / "model" / "params.json"
    if params_path.exists():
        params = json.loads(params_path.read_text())
    elif name == "nominal":
        params = nominal_params(run)
        params_path.write_text(json.dumps(params, indent=2))
        print(f"  {name:<14} reconstructed params -> {params_path}")
    else:
        print(f"  {name:<14} SKIPPED (no params.json)")
        return None

    with np.load(history_path) as data:
        history = {k: data[k] for k in data.files}

    label, description = LABELS[name]
    stats = summarize(history)
    meta = dict(params, ablation=name, label=label, description=description, **stats)
    # params_json rides inside the npz so one file per run is fully sufficient.
    np.savez_compressed(OUT / f"{name}.npz",
                        params_json=np.array(json.dumps(meta)), **history)
    print(f"  {name:<14} {stats.get('episodes', 0):>7} eps | "
          f"dock {stats.get('dock_rate_final', 0):>5.1f}% | "
          f"reward {stats.get('reward_final', 0):>8.1f}")
    return meta


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Collecting into {OUT}")
    manifest = {}
    for name in ORDER:
        meta = collect(name)
        if meta is not None:
            manifest[name] = meta
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(manifest)}/{len(ORDER)} runs -> {OUT / 'manifest.json'}")
    missing = [n for n in ORDER if n not in manifest]
    if missing:
        print(f"Missing: {', '.join(missing)}")


if __name__ == "__main__":
    main()
