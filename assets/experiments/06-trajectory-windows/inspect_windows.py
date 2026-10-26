"""Export the first real K=4 window batch and an original-pixel timeline."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version
import torch
from torch.utils.data import DataLoader

from trajectory_window_dataset import TrajectoryWindowDataset


def flatten(tree, prefix=""):
    result = {}
    for key, value in tree.items():
        name = prefix + key
        if isinstance(value, dict):
            result.update(flatten(value, name + "/"))
        else:
            if value.device.type != "cpu" or value.requires_grad:
                raise ValueError("Only CPU data tensors without gradients are allowed.")
            result[name] = value.numpy().copy()
    return result


def preview(path, rgb, actions, window):
    h, w = rgb.shape[1:3]
    header, label, gap = 56, 32, 16
    row_height = label + h + gap
    image = Image.new("RGB", (max(560, w + 304), header + len(window["state_indices"]) * row_height + gap), "#faf8f4")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=19)
    draw.text((16, 12), f"Window 0 | {len(window['action_indices'])} actions / {len(window['state_indices'])} states", fill="#292b27", font=font)
    tiles = []
    for j, i in enumerate(window["state_indices"]):
        y = header + j * row_height
        draw.text((16, y + 2), f"state {i} | step {window['step_index'][j]} | t={window['time_s'][j]:.2f}s", fill="#292b27", font=font)
        image.paste(Image.fromarray(rgb[i]), (16, y + label))
        tiles.append({"state_index": i, "step_index": window["step_index"][j], "time_s": window["time_s"][j],
                      "box_xywh": [16, y + label, w, h]})
        x = w + 40
        if j < len(window["action_indices"]):
            a = window["action_indices"][j]
            lines = [f"action[{a}] -> state {i+1}", f"q1 input: {actions[a,0]:+.1f}",
                     f"gripper input: {actions[a,7]:+.1f}", "(selected components)",
                     f"dt: {window['time_s'][j+1]-window['time_s'][j]:.2f}s"]
        else:
            lines = ["Final state in window", f"K={len(window['action_indices'])} actions",
                     f"K+1={len(window['state_indices'])} observations", "Raw pixels unchanged"]
        for row, text in enumerate(lines):
            draw.text((x, y + 68 + row * 30), text, fill="#51554d", font=font)
    image.save(path)
    return {"size_wh": list(image.size), "tiles": tiles,
            "note": "Five source RGB tiles pasted without resize/filter/augmentation; labels are outside image tiles and are not model inputs."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, default=Path(__file__).resolve().parent / "source/trajectory.npz")
    parser.add_argument("--output-dir", type=Path, default=Path("result"))
    parser.add_argument("--window-length", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()
    names = ("window_index.json", "sample_batch.npz", "sample_batch.json", "window_preview.png", "schema.json", "environment.json")
    if any((args.output_dir / name).exists() for name in names):
        raise FileExistsError("Preserve existing results; choose a new --output-dir.")
    source_hash = sha256(args.trajectory.read_bytes()).hexdigest()
    dataset = TrajectoryWindowDataset(args.trajectory, K=args.window_length, stride=args.stride)
    if len(dataset) == 0:
        raise ValueError("No complete window: K exceeds the available actions; no padding or partial sample exported.")
    loader = DataLoader(dataset, batch_size=4, shuffle=False, drop_last=False, num_workers=0,
                        pin_memory=False, generator=torch.Generator(device="cpu").manual_seed(42))
    first, last, batches = None, None, []
    for batch in loader:
        flat = flatten(batch)
        batches.append(flat["metadata/window_index"].tolist())
        if first is None:
            first = flat
        last = flat
    if [i for block in batches for i in block] != list(range(len(dataset))):
        raise ValueError("Window indices were lost, repeated or reordered.")
    with np.load(args.trajectory, allow_pickle=False) as saved:
        rgb = saved["obs/sensor_data/base_camera/rgb"][:, 0].copy()
        actions = saved["actions"].copy()
        steps = saved["step_index"][:, 0].copy()
        times = saved["time_s"][:, 0].copy()
    windows = []
    for i, s in enumerate(dataset.starts):
        e = s + dataset.K
        windows.append({"window_index": i, "source_id": 0, "start_index": s, "end_index_exclusive": e,
                        "state_indices": list(range(s, e + 1)), "action_indices": list(range(s, e)),
                        "step_index": steps[s:e+1].tolist(), "time_s": times[s:e+1].tolist(),
                        "ends_fragment": e == dataset.n})
    loader_config = {"batch_size": 4, "shuffle": False, "drop_last": False, "num_workers": 0,
                     "pin_memory": False, "generator_seed": 42, "device": "cpu"}
    index = {"format": "article6-window-index-v1", "K": dataset.K, "stride": dataset.stride,
             "source_actions": dataset.n, "source_observations": len(rgb), "window_count": len(dataset),
             "sources": {"0": {"file": "trajectory.npz", "sha256": source_hash, "environment_index": 0}},
             "policy": "Complete windows only; no padding, terminal continuation, resets, or source-file joining.",
             "windows": windows}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "sample_batch.npz", **first)
    figure = preview(args.output_dir / "window_preview.png", rgb, actions, windows[0])
    fields = {}
    for key, value in first.items():
        if key.startswith("observations/"):
            axes = ["B", "K+1"] + (["C", "H", "W"] if key.endswith("rgb") else ["J"])
            units = "RGB intensity [0,1]" if key.endswith("rgb") else ["rad/s"]*7+["m/s"]*2 if key.endswith("qvel") else ["rad"]*7+["m"]*2
        elif key == "actions":
            axes, units = ["B", "K", "A"], "normalized controller commands [-1,1], unchanged"
        elif key in ("rewards", "terminated", "truncated"):
            axes, units = ["B", "K"], "task reward, not success" if key == "rewards" else "boolean, unchanged episode flag"
        elif key in ("metadata/state_indices", "metadata/step_index", "metadata/time_s"):
            axes, units = ["B", "K+1"], "simulation seconds" if key.endswith("time_s") else "integer state index/control-step count"
        elif key == "metadata/action_indices":
            axes, units = ["B", "K"], "integer source transition index"
        else:
            axes, units = ["B"], "boolean fragment end, not episode end" if key.endswith("ends_fragment") else "integer metadata"
        fields[key] = {"shape": list(value.shape), "dtype": str(value.dtype), "axes": axes, "units": units}
    schema = {"format": "article6-window-batch-v1", "K": dataset.K, "stride": dataset.stride,
              "fields": fields, "environment_selection": "Explicit environment 0 in copied article5 loader; no unqualified squeeze.",
              "observation_subset": ["qpos", "qvel", "rgb"], "excluded_from_inputs": ["diagnostic/*", "info/success", "controller targets", "camera matrices", "tcp_pose"],
              "source_axes": {"qpos_qvel": ["T+1", "E", "J"], "rgb": ["T+1", "E", "H", "W", "C"], "actions": ["T", "A"]},
              "end_index_exclusive": "Exclusive action stop s+K; also final included state index. State slice stop is s+K+1.",
              "causality": "Offline sequence record only. Future states are not assumed visible at window start. Context/targets/visibility must be defined for a future model.",
              "source_id": "Numeric key into sources in window_index.json; identifies a fragment, not a claim of a complete episode.",
              "ends_fragment": "Separate metadata; never substituted for terminated or truncated."}
    summary = {"result": "exported_pending_independent_validation", "source_sha256": source_hash,
               "K": dataset.K, "stride": dataset.stride, "window_count": len(dataset), "loader": loader_config,
               "batch_count": len(batches), "batch_sizes": [len(x) for x in batches], "batch_window_indices": batches,
               "first_window": windows[0], "last_window": windows[-1],
               "source_simulation_duration_s": float(times[-1]-times[0]),
               "window_duration_s": float(times[dataset.K]-times[0]),
               "preview": figure,
               "fields": {k: {"shape": list(v.shape), "dtype": str(v.dtype), "min": v.min().item(), "max": v.max().item()} for k,v in first.items()},
               "last_batch_shapes": {k:list(v.shape) for k,v in last.items()},
               "note": "Overlapping windows do not add independent trajectories or simulation time. No train/validation split or training."}
    helper_path = Path(__file__).resolve().parent / "trajectory_dataset.py"
    environment = {"record_type": "local_offline_window_export", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                   "python_version": platform.python_version(), "packages": {"numpy": np.__version__, "torch": torch.__version__, "pillow": pillow_version},
                   "device": "cpu", "torch_cuda_build": torch.version.cuda, "offline_only": True, "simulator_calls": 0,
                   "training": False, "source_file": "trajectory.npz", "source_sha256": source_hash,
                   "article5_helper": {"file": "trajectory_dataset.py", "sha256": sha256(helper_path.read_bytes()).hexdigest(), "copied_unchanged": True},
                   "configuration": {"K": dataset.K, "stride": dataset.stride, **loader_config},
                   "local_paths": {"python": sys.executable, "source": str(args.trajectory.resolve()), "output": str(args.output_dir.resolve())}}
    for name, value in (("window_index.json", index), ("sample_batch.json", summary), ("schema.json", schema), ("environment.json", environment)):
        (args.output_dir / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if sha256(args.trajectory.read_bytes()).hexdigest() != source_hash:
        raise ValueError("Input changed during export.")
    print(json.dumps({"result": "EXPORTED", "windows": len(dataset), "K": dataset.K, "stride": dataset.stride,
                      "batch_sizes": summary["batch_sizes"], "arrays": len(first), "device": "cpu", "preview_size": figure["size_wh"]}))


if __name__ == "__main__":
    main()
