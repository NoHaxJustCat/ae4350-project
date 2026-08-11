"""
Gather the ablation study into one self-contained directory for plotting.

    python scripts/collect_ablations.py

Reads tmp/abl_*/model/{history.npz,params.json} and writes for each run:

    out/ablations/<name>.npz      per-episode history + a "params_json" entry

so a plotting script needs exactly one file per run and never has to consult
config.py, which will have moved on by the time the report is written. A
human-readable out/ablations/manifest.json indexes them.

Output goes to out/, not tmp/: tmp/ is gitignored scratch that gets cleared,
and the collected study is what the report is written from.

The nominal is tmp/abl_nominal -- a real seeded run at the same settings as the
rest, NOT the older tmp/vbar_fresh. See runs/ablations.ps1 for why that one
cannot serve as a baseline.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import OUT_DIR

# out/ is tracked and tmp/ is gitignored, and the collected study is the thing
# the report is written from, so it belongs in out/.
OUT = Path(__file__).resolve().parents[1] / "out" / "ablations"

# What each run is meant to show, carried into the manifest so the plots can be
# labelled without re-deriving it from the parameter diff.
LABELS = {
    "nominal":       ("Nominal", "TD3, gamma 0.99, lr 1e-4, net [256,256], replay 300k"),
    "gamma_095":     ("gamma 0.95", "Discount lowered from 0.99"),
    "gamma_090":     ("gamma 0.90", "Discount lowered from 0.99"),
    "gamma_080":     ("gamma 0.80", "Discount lowered from 0.99"),
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


def run_dir_for(name: str) -> Path:
    return Path(OUT_DIR) / f"abl_{name}"


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
    if not params_path.exists():
        print(f"  {name:<14} SKIPPED (no params.json)")
        return None
    params = json.loads(params_path.read_text())

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
