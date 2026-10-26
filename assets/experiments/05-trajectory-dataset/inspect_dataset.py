"""Export a real first batch and original-pixel preview, entirely on CPU."""
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

from trajectory_dataset import TrajectoryDataset


def flatten_tensors(tree, prefix=""):
    result = {}
    for key, value in tree.items():
        name = prefix + key
        if isinstance(value, dict):
            result.update(flatten_tensors(value, name + "/"))
        else:
            if value.device.type != "cpu" or value.requires_grad:
                raise ValueError("Only independent CPU data tensors are allowed.")
            result[name] = value.numpy().copy()
    return result


def make_preview(path, raw_rgb, indices, steps, times):
    """Paste unmodified 256x256 source images; labels occupy separate margins."""
    height, width = raw_rgb.shape[1:3]
    margin, gap, heading, label, row_gap = 16, 16, 56, 34, 18
    canvas = Image.new("RGB", (2 * width + 2 * margin + gap,
                                heading + len(indices) * (label + height + row_gap)), "#faf8f4")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=17)
    draw.text((margin, 8), "First batch: 4 real transitions", fill="#292b27", font=font)
    draw.text((margin, 31), "obs[i]", fill="#51554d", font=font)
    draw.text((margin + width + gap, 31), "next_obs[i] = obs[i+1]", fill="#51554d", font=font)
    tiles = []
    for row, i in enumerate(indices):
        y = heading + row * (label + height + row_gap)
        text = f"sample {i} | step {steps[i]} -> {steps[i+1]} | {times[i]:.2f}s -> {times[i+1]:.2f}s"
        draw.text((margin, y + 4), text, fill="#292b27", font=font)
        for column, state_index in enumerate((i, i + 1)):
            x, top = margin + column * (width + gap), y + label
            canvas.paste(Image.fromarray(raw_rgb[state_index]), (x, top))
            tiles.append({"sample_index": int(i), "state_index": int(state_index),
                          "role": "obs" if column == 0 else "next_obs",
                          "box_xywh": [x, top, width, height]})
    canvas.save(path)
    return {"size_wh": list(canvas.size), "tiles": tiles,
            "note": "Original RGB pixels pasted without resizing, filtering, or color conversion; labels outside tiles."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path,
                        default=Path(__file__).resolve().parent / "source/trajectory.npz")
    parser.add_argument("--output-dir", type=Path, default=Path("result"))
    args = parser.parse_args()
    outputs = ("sample_batch.npz", "sample_batch.json", "sample_preview.png", "schema.json", "environment.json")
    if any((args.output_dir / name).exists() for name in outputs):
        raise FileExistsError("Preserve existing exports; choose a new --output-dir.")
    digest_before = sha256(args.trajectory.read_bytes()).hexdigest()
    dataset = TrajectoryDataset(args.trajectory)
    generator = torch.Generator(device="cpu").manual_seed(42)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, drop_last=False,
                        num_workers=0, pin_memory=False, generator=generator)
    first = None
    batch_indices = []
    for batch in loader:
        flat = flatten_tensors(batch)
        batch_indices.append(flat["metadata/sample_index"].tolist())
        if first is None:
            first = flat
    if first is None:
        raise ValueError("No batch was produced.")
    if [i for indices in batch_indices for i in indices] != list(range(len(dataset))):
        raise ValueError("Batches omitted, repeated, or reordered samples.")
    with np.load(args.trajectory, allow_pickle=False) as archive:
        raw_rgb = archive["obs/sensor_data/base_camera/rgb"][:, 0].copy()
        steps = archive["step_index"][:, 0].copy()
        times = archive["time_s"][:, 0].copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "sample_batch.npz", **first)
    preview = make_preview(args.output_dir / "sample_preview.png", raw_rgb,
                           batch_indices[0], steps, times)
    fields = {}
    for key, value in first.items():
        unit = "dimensionless"
        if key.endswith("qpos"):
            unit = ["rad"] * 7 + ["m"] * 2
        elif key.endswith("qvel"):
            unit = ["rad/s"] * 7 + ["m/s"] * 2
        elif key.endswith("time_s"):
            unit = "s (simulation time, not loading wall time)"
        elif key.endswith("rgb"):
            unit = "normalized RGB intensity [0,1]"
        elif key in ("terminated", "truncated", "metadata/is_last_sample"):
            unit = "boolean"
        elif key == "reward":
            unit = "task reward; not task success"
        axes = ["B"] + (["C", "H", "W"] if key.endswith("rgb")
                          else ["J"] if key.endswith(("qpos", "qvel"))
                          else ["A"] if key == "action" else [])
        fields[key] = {"shape": list(value.shape), "dtype": str(value.dtype),
                       "sample_shape": list(value.shape[1:]), "axes": axes, "units": unit}
    schema = {
        "format": "article5-transition-batch-v1", "fields": fields,
        "source_axes": {"qpos_qvel": ["T+1", "E", "J"], "rgb": ["T+1", "E", "H", "W", "C"],
                        "actions": ["T", "A"], "reward_flags": ["T", "E"]},
        "selection": "Explicit environment 0; obs[i], action[i], obs[i+1]. B is not a sequence-model time axis.",
        "observation_subset": ["qpos", "qvel", "rgb"],
        "excluded_from_inputs": ["camera matrices", "tcp_pose", "diagnostic/*", "info/success", "reset_info/*"],
        "action_meaning": {"dimensions_0_to_6": "Normalized [-1,1]; joint delta target_rad = 0.1*a relative to measured pre-step qpos.",
                           "dimension_7": "Normalized absolute mimic-gripper target_m = 0.015 + 0.025*a for each finger; a=1 holds target 0.04m."},
        "is_last_sample": "Fragment-end metadata only; never substituted for terminated or truncated.",
    }
    summary = {
        "result": "exported_pending_independent_validation", "source_file": "trajectory.npz",
        "source_sha256": digest_before, "dataset_length": len(dataset), "source_observations": len(raw_rgb),
        "loader": {"batch_size": 4, "shuffle": False, "drop_last": False, "num_workers": 0,
                   "pin_memory": False, "generator_seed": 42, "device": "cpu"},
        "batch_count": len(batch_indices), "batch_indices": batch_indices,
        "fields": {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                           "min": value.min().item(), "max": value.max().item()} for key, value in first.items()},
        "source_rgb_min": int(raw_rgb.min()), "source_rgb_max": int(raw_rgb.max()),
        "source_time_range_s": [float(times[0]), float(times[-1])],
        "preview": preview,
        "last_sample": {"index": len(dataset) - 1,
                        "terminated": bool(dataset[len(dataset) - 1]["terminated"]),
                        "truncated": bool(dataset[len(dataset) - 1]["truncated"]),
                        "is_last_sample": True},
    }
    environment = {
        "record_type": "local_offline_dataset_export", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "packages": {"numpy": np.__version__, "torch": torch.__version__, "pillow": pillow_version},
        "device": "cpu", "torch_cuda_build": torch.version.cuda,
        "offline_only": True, "simulator_calls": 0, "training": False,
        "source_file": "trajectory.npz", "source_sha256": digest_before,
        "local_paths": {"python": sys.executable, "source": str(args.trajectory.resolve()),
                        "output": str(args.output_dir.resolve())},
        "loader": summary["loader"],
    }
    for name, content in (("sample_batch.json", summary), ("schema.json", schema), ("environment.json", environment)):
        (args.output_dir / name).write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if sha256(args.trajectory.read_bytes()).hexdigest() != digest_before:
        raise ValueError("Input file changed during export.")
    print(json.dumps({"result": "EXPORTED", "samples": len(dataset), "batches": len(batch_indices),
                      "first_indices": batch_indices[0], "arrays": len(first), "device": "cpu",
                      "preview_size": preview["size_wh"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
