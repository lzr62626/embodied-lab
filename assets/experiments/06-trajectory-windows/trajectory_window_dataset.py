"""One saved trajectory -> complete, independently stored CPU time windows."""
from operator import index as integer_index

import numpy as np
import torch
from torch.utils.data import Dataset

from trajectory_dataset import TrajectoryDataset


def positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer, not a boolean.")
    try:
        value = integer_index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be a positive integer.") from error
    if value < 1:
        raise ValueError(f"{name} must be at least 1.")
    return value


class TrajectoryWindowDataset(Dataset):
    """K actions and K+1 states per window; no padding or episode joining.

    Reuses article 5's exact loader. It validates a single continuous fragment,
    rejects any action after a terminal/truncated transition, selects env 0,
    and copies qpos/qvel/RGB into CPU tensors. An end flag at the final source
    transition is valid. A file fragment may also end with both flags false.
    """

    def __init__(self, trajectory_path, K=4, stride=1):
        self.K = positive_integer(K, "K")
        self.stride = positive_integer(stride, "stride")
        self.base = TrajectoryDataset(trajectory_path)
        self.n = len(self.base)
        self.starts = tuple(range(0, max(0, self.n - self.K + 1), self.stride))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        w = integer_index(index)
        if w < 0 or w >= len(self):
            raise IndexError(f"Window index {w} is outside [0,{len(self)}).")
        start = self.starts[w]
        stop = start + self.K  # action-exclusive stop == final state index
        transitions = [self.base[i] for i in range(start, stop)]
        states = [sample["obs"] for sample in transitions] + [transitions[-1]["next_obs"]]
        # stack allocates new storage, including where adjacent windows overlap.
        observations = {key: torch.stack([state[key] for state in states], dim=0)
                        for key in ("qpos", "qvel", "rgb")}
        steps = [sample["metadata"]["step_index"] for sample in transitions]
        steps.append(transitions[-1]["metadata"]["next_step_index"])
        times = [sample["metadata"]["time_s"] for sample in transitions]
        times.append(transitions[-1]["metadata"]["next_time_s"])
        return {
            "observations": observations,
            "actions": torch.stack([sample["action"] for sample in transitions]),
            "rewards": torch.stack([sample["reward"] for sample in transitions]),
            "terminated": torch.stack([sample["terminated"] for sample in transitions]),
            "truncated": torch.stack([sample["truncated"] for sample in transitions]),
            "metadata": {
                "source_id": torch.tensor(0, dtype=torch.int64, device="cpu"),
                "window_index": torch.tensor(w, dtype=torch.int64, device="cpu"),
                "start_index": torch.tensor(start, dtype=torch.int64, device="cpu"),
                "end_index_exclusive": torch.tensor(stop, dtype=torch.int64, device="cpu"),
                "state_indices": torch.arange(start, stop + 1, dtype=torch.int64, device="cpu"),
                "action_indices": torch.arange(start, stop, dtype=torch.int64, device="cpu"),
                "step_index": torch.stack(steps),
                "time_s": torch.stack(times),
                "environment_index": torch.tensor(0, dtype=torch.int64, device="cpu"),
                "ends_fragment": torch.tensor(stop == self.n, dtype=torch.bool, device="cpu"),
            },
        }
