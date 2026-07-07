import numpy as np
import torch
from libs.constants import ENV_BOUNDARY, ENV_MAX_DV, ENV_DV_BUDGET, OMEGA
STATE_SCALE = np.array(
    [
        ENV_BOUNDARY,
        ENV_BOUNDARY,
        ENV_BOUNDARY,
        OMEGA * ENV_BOUNDARY,
        OMEGA * ENV_BOUNDARY,
        OMEGA * ENV_BOUNDARY,
        ENV_DV_BUDGET,
    ],
    dtype=np.float64,
)
ACTION_SCALE = np.array([ENV_MAX_DV, ENV_MAX_DV, ENV_MAX_DV], dtype=np.float64)
def _normalize_array(values: np.ndarray, scale: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -scale, scale)
    return (clipped + scale) / (2.0 * scale)
def _normalize_tensor(values: torch.Tensor, scale: np.ndarray) -> torch.Tensor:
    scale_tensor = torch.as_tensor(scale, dtype=values.dtype, device=values.device)
    clipped = torch.clamp(values, -scale_tensor, scale_tensor)
    return (clipped + scale_tensor) / (2.0 * scale_tensor)
def normalize_state(state):
    if isinstance(state, torch.Tensor):
        return _normalize_tensor(state, STATE_SCALE)
    return _normalize_array(np.asarray(state, dtype=np.float64), STATE_SCALE)
def normalize_action(action):
    if isinstance(action, torch.Tensor):
        return _normalize_tensor(action, ACTION_SCALE)
    return _normalize_array(np.asarray(action, dtype=np.float64), ACTION_SCALE)