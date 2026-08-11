"""Output layout for a run.

    tmp/<run_tag>/
        episodes/                 per-episode trajectory plots
        model/
            checkpoints/          periodic checkpoints + curriculum sidecars
            <scenario>_td3.zip    final model
            best_model.zip        best deterministic evaluation
            status.json
        diagnostics/
            plots/                one figure per metric
"""

import shutil
from pathlib import Path

from config import OUT_DIR


class RunPaths:
    def __init__(self, run_tag: str, scenario: str, clean: bool = True):
        self.run_tag = run_tag
        self.scenario = scenario
        self.root = Path(OUT_DIR) / run_tag
        if clean and self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)
        for d in (self.episodes, self.checkpoints, self.panels):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def episodes(self) -> Path:
        return self.root / "episodes"

    @property
    def model(self) -> Path:
        return self.root / "model"

    @property
    def checkpoints(self) -> Path:
        return self.model / "checkpoints"

    @property
    def diagnostics(self) -> Path:
        return self.root / "diagnostics"

    @property
    def panels(self) -> Path:
        return self.diagnostics / "plots"

    @property
    def final_model(self) -> Path:
        return self.model / f"{self.scenario}_td3"

    @property
    def best_model(self) -> Path:
        return self.model / "best_model"

    @property
    def status(self) -> Path:
        return self.model / "status.json"

    @property
    def history(self) -> Path:
        return self.model / "history.npz"

    @property
    def params(self) -> Path:
        return self.model / "params.json"
