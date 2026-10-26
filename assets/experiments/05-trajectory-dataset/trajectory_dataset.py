"""One saved single-environment trajectory -> independent CPU transitions.

Only the numeric NPZ is required. No simulator, model, or GPU is used.
"""
from operator import index as integer_index
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """Article-4 format: T+1 states, T actions, exactly one environment.

    Each item selects environment 0 explicitly. The observation is a documented
    subset (qpos, qvel, RGB), not every field in the source policy observation.
    Diagnostic simulator truth, controller targets and success are not loaded.
    """

    REQUIRED = (
        "obs/agent/qpos", "obs/agent/qvel", "obs/sensor_data/base_camera/rgb",
        "actions", "rewards", "terminated", "truncated", "step_index", "time_s",
    )

    def __init__(self, trajectory_path):
        self.path = Path(trajectory_path)
        with np.load(self.path, allow_pickle=False) as archive:
            missing = set(self.REQUIRED) - set(archive.files)
            if missing:
                raise ValueError(f"Missing trajectory fields: {sorted(missing)}")
            self._arrays = {key: archive[key].copy() for key in self.REQUIRED}
        a = self._arrays
        if a["actions"].ndim != 2 or a["actions"].shape[1] != 8:
            raise ValueError("Expected actions shaped (T,8), with no environment axis.")
        self.n = len(a["actions"])
        self.environment_index = 0
        if self.n < 1:
            raise ValueError("The trajectory must contain at least one transition.")
        expected = {
            "obs/agent/qpos": ((self.n + 1, 1, 9), np.float32),
            "obs/agent/qvel": ((self.n + 1, 1, 9), np.float32),
            "actions": ((self.n, 8), np.float32),
            "rewards": ((self.n, 1), np.float32),
            "terminated": ((self.n, 1), np.bool_),
            "truncated": ((self.n, 1), np.bool_),
            "step_index": ((self.n + 1, 1), np.int32),
            "time_s": ((self.n + 1, 1), np.float64),
        }
        for key, (shape, dtype) in expected.items():
            if a[key].shape != shape or a[key].dtype != np.dtype(dtype):
                raise ValueError(f"Unexpected shape or dtype for {key}: {a[key].shape}, {a[key].dtype}")
        rgb = a["obs/sensor_data/base_camera/rgb"]
        if (rgb.ndim != 5 or rgb.shape[:2] != (self.n + 1, 1)
                or rgb.shape[-1] != 3 or min(rgb.shape[2:4]) < 1 or rgb.dtype != np.uint8):
            raise ValueError("Expected RGB uint8 (T+1,1,H,W,3).")
        if not all(np.isfinite(value).all() for value in a.values()):
            raise ValueError("Trajectory contains non-finite values.")
        if np.any(np.abs(a["actions"]) > 1):
            raise ValueError("Normalized actions must lie in [-1,1].")
        if not np.all(np.diff(a["step_index"][:, 0]) == 1):
            raise ValueError("Control-step indices must be consecutive.")
        if not np.all(np.diff(a["time_s"][:, 0]) > 0):
            raise ValueError("Simulation time must increase.")
        if a["terminated"][:-1].any() or a["truncated"][:-1].any():
            raise ValueError("This single-segment loader cannot cross an episode boundary.")
        for value in a.values():
            value.setflags(write=False)

    def __len__(self):
        return self.n

    @staticmethod
    def _tensor(value):
        # from_numpy shares storage with this NEW copy, never the source cache.
        return torch.from_numpy(np.array(value, copy=True))

    def _observation(self, state_index):
        a, e = self._arrays, self.environment_index
        # Explicit [state_index, environment_index], never an unqualified squeeze.
        hwc = a["obs/sensor_data/base_camera/rgb"][state_index, e]
        chw = hwc.transpose(2, 0, 1).copy()
        rgb = torch.from_numpy(chw).to(device="cpu", dtype=torch.float32).div_(255.0)
        return {
            "qpos": self._tensor(a["obs/agent/qpos"][state_index, e]),
            "qvel": self._tensor(a["obs/agent/qvel"][state_index, e]),
            "rgb": rgb,
        }

    def __getitem__(self, index):
        i = integer_index(index)
        if i < 0 or i >= self.n:
            raise IndexError(f"Sample index {i} is outside [0,{self.n}).")
        a, e = self._arrays, self.environment_index
        return {
            "obs": self._observation(i),
            "action": self._tensor(a["actions"][i]),
            "next_obs": self._observation(i + 1),
            "reward": self._tensor(a["rewards"][i, e]),
            "terminated": self._tensor(a["terminated"][i, e]),
            "truncated": self._tensor(a["truncated"][i, e]),
            "metadata": {
                "sample_index": torch.tensor(i, dtype=torch.int64, device="cpu"),
                "step_index": torch.tensor(int(a["step_index"][i, e]), dtype=torch.int64, device="cpu"),
                "next_step_index": torch.tensor(int(a["step_index"][i + 1, e]), dtype=torch.int64, device="cpu"),
                "time_s": self._tensor(a["time_s"][i, e]),
                "next_time_s": self._tensor(a["time_s"][i + 1, e]),
                "environment_index": torch.tensor(e, dtype=torch.int64, device="cpu"),
                # Segment end is metadata; it does NOT rewrite either episode flag.
                "is_last_sample": torch.tensor(i == self.n - 1, dtype=torch.bool, device="cpu"),
            },
        }
