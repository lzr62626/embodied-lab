"""Capture exactly one initialized ManiSkill sensor RGB frame (no env.step).

Validated versions and results are in environment.json and README.md.
The CPU rendering backend means CPU-accessible image transfer; SAPIEN still
uses the GPU's Vulkan renderer. This script does not use CUDA camera tensors.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def installed_versions() -> dict:
    result = {}
    for name in ("mani-skill", "sapien", "torch", "gymnasium", "numpy", "pillow"):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def gpu_info() -> dict:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"available": False}
    result = subprocess.run(
        [executable, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=20,
    )
    return {"exit_code": result.returncode, "description": result.stdout.strip(),
            "stderr": result.stderr.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    image_path, report_path = output / "frame.png", output / "environment.json"
    if image_path.exists() or report_path.exists():
        parser.error("Output already exists. Choose a new --output-dir to preserve previous results.")

    # Keep all newly downloaded assets/caches in the task workspace, not in ~.
    cache_root = Path(os.environ.get("SIM_CAMERA_CACHE", str(output / ".cache"))).resolve()
    for variable, folder in (("MS_ASSET_DIR", "maniskill-assets"),
                             ("HF_HOME", "huggingface"), ("TORCH_HOME", "torch")):
        target = cache_root / folder
        target.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(variable, str(target))
    os.environ.setdefault("MS_SKIP_ASSET_DOWNLOAD_PROMPT", "1")
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    started = time.perf_counter()
    env = None
    report = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "initializing", "platform": platform.platform(),
        "windows_version": platform.win32_ver(),
        "python": {"executable": sys.executable, "version": sys.version},
        "packages": installed_versions(), "gpu": gpu_info(),
        "experiment": {
            "env_id": "PushCube-v1", "robot_uids": "panda", "num_envs": 1,
            "seed": args.seed, "obs_mode": "rgb", "camera_name": "base_camera",
            "camera_role": "sensor observation, available as policy input; no policy was run",
            "sim_backend": "cpu", "render_backend": "cpu",
            "rendering": "SAPIEN Vulkan GPU renderer with CPU-accessible image transfer",
            "control_mode": "pd_joint_delta_pos", "render_mode": "rgb_array",
            "sensor_configs": {"width": 256, "height": 256, "shader_pack": "minimal"},
            "explicit_reset_calls": 0, "step_calls": 0, "frames_saved": 0, "training": False,
            "constructor_note": "ManiSkill 3.0.1 also resets internally while constructing the environment.",
        },
        "cache_paths": {key: os.environ[key] for key in ("MS_ASSET_DIR", "HF_HOME", "TORCH_HOME")},
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    try:
        import gymnasium as gym
        import mani_skill.envs  # Registers ManiSkill task IDs with Gymnasium.
        import numpy as np
        import torch
        from PIL import Image
        from mani_skill.envs.tasks.tabletop import push_cube

        source = Path(push_cube.__file__).resolve()
        report["provenance"] = {"task_source": str(source), "task_source_sha256": sha256(source),
                                "torch_cuda_build": torch.version.cuda,
                                "capture_method": "env.reset -> obs['sensor_data']['base_camera']['rgb'][0]"}
        print("Creating one PushCube-v1 environment (CPU physics / Vulkan rendering).", flush=True)
        env = gym.make(
            "PushCube-v1", robot_uids="panda", num_envs=1, obs_mode="rgb",
            sim_backend="cpu", render_backend="cpu", control_mode="pd_joint_delta_pos",
            render_mode="rgb_array", sensor_configs=report["experiment"]["sensor_configs"],
        )
        observation, _ = env.reset(seed=args.seed)
        report["experiment"]["explicit_reset_calls"] = 1
        backend = env.unwrapped.backend
        report["resolved_backend"] = {
            "sim_backend": backend.sim_backend, "render_backend": backend.render_backend,
            "tensor_device": str(backend.device), "sim_device": str(backend.sim_device),
            "render_device": str(backend.render_device),
        }
        report["camera_parameters"] = {
            key: value.detach().cpu().tolist()
            for key, value in observation.get("sensor_param", {}).get("base_camera", {}).items()
        }
        sensors = observation["sensor_data"]
        if "base_camera" not in sensors:
            raise RuntimeError(f"base_camera missing; available cameras: {list(sensors)}")
        batch = sensors["base_camera"]["rgb"]
        report["observation"] = {"key": "sensor_data/base_camera/rgb",
                                 "batched_shape": list(batch.shape), "dtype": str(batch.dtype),
                                 "device": str(batch.device)}
        rgb = batch[0].detach().cpu().numpy()
        if rgb.shape != (256, 256, 3) or rgb.dtype != np.uint8:
            raise RuntimeError(f"Unexpected sensor frame: {rgb.shape}, {rgb.dtype}")
        if not np.isfinite(rgb).all() or float(rgb.std()) < 1.0:
            raise RuntimeError("Frame is invalid or nearly uniform.")
        Image.fromarray(rgb).save(image_path, format="PNG")
        with Image.open(image_path) as reopened:
            reopened.load()
            saved = np.asarray(reopened)
            if reopened.mode != "RGB" or not np.array_equal(rgb, saved):
                raise RuntimeError("Saved PNG does not match the original RGB observation.")
        report["experiment"]["frames_saved"] = 1
        report["image"] = {
            "file": "frame.png", "width": int(rgb.shape[1]), "height": int(rgb.shape[0]),
            "channels": 3, "dtype": str(rgb.dtype), "pixel_min": int(rgb.min()),
            "pixel_max": int(rgb.max()), "std": float(rgb.std()),
            "unique_rgb_colors": int(len(np.unique(rgb.reshape(-1, 3), axis=0))),
            "png_reopened_equal_to_sensor": True, "sha256": sha256(image_path),
        }
        report["status"] = "capture_passed_pending_visual_review"
        print(json.dumps(report["image"], ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error),
                           "traceback": traceback.format_exc()}
        traceback.print_exc()
        return 1
    finally:
        if env is not None:
            env.close()
        report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Report: {report_path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
