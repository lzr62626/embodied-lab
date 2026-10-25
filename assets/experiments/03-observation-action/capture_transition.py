"""Save one real PushCube-v1 transition; exactly one env.step, no training.

Uses CPU physics and CPU-accessible SAPIEN/Vulkan image transfer. Existing
results are never overwritten. Package versions and runtime config are saved.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import sys
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "result")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("environment.json", "transition.npz", "before.png", "after.png")):
        parser.error("Results exist. Choose a fresh --output-dir to preserve them.")
    cache = Path(os.environ.get("SIM_CAMERA_CACHE", str(output / ".cache"))).resolve()
    for key, folder in (("MS_ASSET_DIR", "maniskill-assets"), ("HF_HOME", "huggingface"),
                        ("TORCH_HOME", "torch"), ("MPLCONFIGDIR", "matplotlib")):
        location = cache / folder
        location.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(key, str(location))
    os.environ.setdefault("MS_SKIP_ASSET_DOWNLOAD_PROMPT", "1")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    report = {
        "status": "started", "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "packages": {name: metadata.version(name) for name in
                     ("mani-skill", "sapien", "torch", "gymnasium", "numpy", "pillow")},
        "experiment": {"task": "PushCube-v1", "robot": "panda", "num_envs": 1,
                       "seed": args.seed, "obs_mode": "rgb", "camera": "base_camera",
                       "camera_role": "policy_observation_sensor", "width": 256, "height": 256,
                       "shader_pack": "minimal", "sim_backend": "cpu", "render_backend": "cpu",
                       "control_mode": "pd_joint_delta_pos", "render_mode": "rgb_array",
                       "explicit_reset_calls": 0, "step_calls": 0, "training": False},
        "local_paths": {"python_executable": sys.executable, "cache": str(cache)},
        "script_sha256": sha(__file__),
    }
    env = None
    try:
        import gymnasium as gym
        import mani_skill.envs  # Register tasks.
        import numpy as np
        import torch
        from PIL import Image

        def array(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().numpy().copy()
            return np.array(value, copy=True)

        # Flatten dictionary paths, never dictionary objects, into safe numeric NPZ entries.
        def snapshot(value, prefix, arrays, fields, empty):
            if isinstance(value, dict):
                if not value:
                    empty.append(prefix)
                for key, child in value.items():
                    snapshot(child, prefix + "/" + key, arrays, fields, empty)
                return
            copied = array(value)
            if copied.dtype.hasobject or copied.dtype.kind not in "buifc":
                raise TypeError("Non-numeric field: " + prefix)
            if not np.isfinite(copied).all():
                raise ValueError("Non-finite field: " + prefix)
            arrays[prefix] = copied
            fields[prefix] = {"shape": list(copied.shape), "dtype": str(copied.dtype),
                              "source_type": type(value).__name__,
                              "source_device": str(value.device) if isinstance(value, torch.Tensor) else None}

        def image(name, batch):
            rgb = batch[0]
            if batch.shape != (1, 256, 256, 3) or rgb.dtype != np.uint8:
                raise ValueError("Unexpected RGB observation shape/dtype")
            Image.fromarray(rgb).save(output / name)
            with Image.open(output / name) as reopened:
                reopened.load()
                if reopened.mode != "RGB" or not np.array_equal(np.asarray(reopened), rgb):
                    raise ValueError("PNG does not equal sensor RGB")
            return {"width": 256, "height": 256, "channels": 3, "dtype": str(rgb.dtype),
                    "pixel_min": int(rgb.min()), "pixel_max": int(rgb.max()),
                    "sha256": sha(output / name), "png_equals_sensor": True}

        env = gym.make("PushCube-v1", robot_uids="panda", num_envs=1, obs_mode="rgb",
                       sim_backend="cpu", render_backend="cpu", control_mode="pd_joint_delta_pos",
                       render_mode="rgb_array", sensor_configs={"width": 256, "height": 256,
                                                                "shader_pack": "minimal"})
        obs, info = env.reset(seed=args.seed)
        report["experiment"]["explicit_reset_calls"] = 1
        base = env.unwrapped
        controller = base.agent.controller
        report["resolved_backend"] = {key: str(getattr(base.backend, key)) for key in
                                      ("sim_backend", "render_backend", "device", "sim_device", "render_device")}
        report["timing"] = {"control_freq_hz": base.control_freq, "simulation_freq_hz": base.sim_freq,
                            "control_dt_s": base.control_timestep, "simulation_dt_s": base.sim_timestep,
                            "physics_substeps_per_control": base._sim_steps_per_control}
        report["action_space"] = {"shape": list(env.action_space.shape), "dtype": str(env.action_space.dtype),
                                  "low": env.action_space.low.tolist(), "high": env.action_space.high.tolist()}
        report["controllers"] = {
            name: {"class": type(part).__name__, "config": asdict(part.config),
                   "joint_names": [joint.name for joint in part.joints],
                   "active_joint_indices": array(part.active_joint_indices).tolist(),
                   "physical_low": part._original_single_action_space.low.tolist(),
                   "physical_high": part._original_single_action_space.high.tolist(),
                   "action_slice": list(controller.action_mapping[name])}
            for name, part in controller.controllers.items()
        }
        arm, gripper = controller.controllers["arm"], controller.controllers["gripper"]
        # Fail before acting if the installed controller differs from the checked implementation.
        assert controller.action_mapping == {"arm": (0, 7), "gripper": (7, 8)}
        assert arm.config.normalize_action and arm.config.use_delta and not arm.config.use_target
        assert gripper.config.normalize_action and not gripper.config.use_delta
        assert np.allclose(arm._original_single_action_space.low, -0.1)
        assert np.allclose(arm._original_single_action_space.high, 0.1)
        assert np.allclose(gripper._original_single_action_space.low, -0.01)
        assert np.allclose(gripper._original_single_action_space.high, 0.04)

        action = np.array([0.2, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
        assert env.action_space.contains(action) and np.isfinite(action).all()
        arrays, fields, empty = {}, {}, []
        snapshot(obs, "obs_t", arrays, fields, empty)
        snapshot(info, "info_t", arrays, fields, empty)
        snapshot(base.obj.pose.raw_pose, "diagnostic/cube_pose_t", arrays, fields, empty)
        arrays["action"] = action.copy()
        arrays["step_index_t"] = array(base.elapsed_steps)
        arrays["time_t_s"] = arrays["step_index_t"].astype(np.float64) * base.control_timestep
        before_fingerprints = {key: hashlib.sha256(value.tobytes()).hexdigest()
                               for key, value in arrays.items() if key.startswith("obs_t/")}
        report["images"] = {"before.png": image("before.png", arrays["obs_t/sensor_data/base_camera/rgb"])}
        print("Snapshot t copied. Calling env.step exactly once with " + str(action.tolist()), flush=True)

        # The only action execution in this script. No loop, action repeat, or second environment.
        report["experiment"]["step_calls"] += 1
        next_obs, reward, terminated, truncated, next_info = env.step(action.copy())

        snapshot(next_obs, "obs_tp1", arrays, fields, empty)
        snapshot(next_info, "info_tp1", arrays, fields, empty)
        snapshot(base.obj.pose.raw_pose, "diagnostic/cube_pose_tp1", arrays, fields, empty)
        for name, part in controller.controllers.items():
            snapshot(part._target_qpos, "diagnostic/" + name + "_target_qpos", arrays, fields, empty)
        arrays.update(reward=array(reward), terminated=array(terminated), truncated=array(truncated),
                      step_index_tp1=array(base.elapsed_steps))
        arrays["time_tp1_s"] = arrays["step_index_tp1"].astype(np.float64) * base.control_timestep
        np.savez_compressed(output / "transition.npz", **arrays)
        assert all(hashlib.sha256(arrays[key].tobytes()).hexdigest() == digest
                   for key, digest in before_fingerprints.items()), "t snapshot changed after step"
        report["images"]["after.png"] = image("after.png", arrays["obs_tp1/sensor_data/base_camera/rgb"])
        with np.load(output / "transition.npz", allow_pickle=False) as saved:
            assert set(saved.files) == set(arrays)
            assert all(np.array_equal(saved[key], value) for key, value in arrays.items())

        q0, q1 = arrays["obs_t/agent/qpos"], arrays["obs_tp1/agent/qpos"]
        rgb0, rgb1 = arrays["obs_t/sensor_data/base_camera/rgb"], arrays["obs_tp1/sensor_data/base_camera/rgb"]
        difference = np.abs(rgb1.astype(np.int16) - rgb0.astype(np.int16))
        cube_delta = arrays["diagnostic/cube_pose_tp1"][:, :3] - arrays["diagnostic/cube_pose_t"][:, :3]
        expected_arm = q0[:, :7] + action[None, :7] * 0.1
        assert np.allclose(arrays["diagnostic/arm_target_qpos"], expected_arm, atol=1e-7)
        assert np.allclose(arrays["diagnostic/gripper_target_qpos"], 0.04, atol=1e-7)
        transition = {
            "format": "numeric_npz_dictionary_paths_v1", "step_calls": 1,
            "sequence": ["obs_t", "action", "obs_tp1"],
            "action": action.tolist(), "arm_target_delta_rad": (action[:7] * 0.1).tolist(),
            "gripper_target_per_finger_m": 0.04,
            "joint_order": [joint.name for joint in base.agent.robot.get_active_joints()],
            "fields": fields, "empty_dict_paths": empty,
            "npz_arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                           for key, value in arrays.items()},
            "snapshot_is_independent_copy": True, "snapshot_t_unchanged_after_step": True,
            "npz_reopened_equal": True, "time_t_s": arrays["time_t_s"].tolist(),
            "time_tp1_s": arrays["time_tp1_s"].tolist(),
            "qpos_t": q0.tolist(), "qpos_tp1": q1.tolist(), "qpos_delta": (q1 - q0).tolist(),
            "arm_target_qpos": arrays["diagnostic/arm_target_qpos"].tolist(),
            "gripper_target_qpos": arrays["diagnostic/gripper_target_qpos"].tolist(),
            "tcp_translation_delta_m": (arrays["obs_tp1/extra/tcp_pose"][:, :3] - arrays["obs_t/extra/tcp_pose"][:, :3]).tolist(),
            "cube_translation_delta_m": cube_delta.tolist(),
            "cube_translation_norm_m": float(np.linalg.norm(cube_delta)),
            "image_mean_absolute_difference": float(difference.mean()),
            "image_max_absolute_difference": int(difference.max()),
            "changed_pixels": int(np.any(difference != 0, axis=-1).sum()),
            "total_pixels": int(np.prod(rgb0.shape[:-1])),
            "reward": arrays["reward"].tolist(), "terminated": arrays["terminated"].tolist(),
            "truncated": arrays["truncated"].tolist(), "success": arrays["info_tp1/success"].tolist(),
            "diagnostic_note": "cube poses and controller targets are simulator-only diagnostics, not additional fields in the policy RGB observation.",
        }
        write_json(output / "transition.json", transition)
        pkg = Path(metadata.distribution("mani-skill").locate_file("mani_skill"))
        report["source_sha256"] = {name: sha(pkg / name) for name in
                                    ("agents/robots/panda/panda.py", "agents/controllers/base_controller.py",
                                     "agents/controllers/pd_joint_pos.py", "utils/gym_utils.py",
                                     "envs/tasks/tabletop/push_cube.py", "envs/sapien_env.py")}
        report["artifacts_sha256"] = {name: sha(output / name) for name in
                                       ("before.png", "after.png", "transition.npz", "transition.json")}
        report["torch_cuda_build"] = torch.version.cuda
        report["status"] = "capture_passed_pending_independent_validation"
        print(json.dumps({key: transition[key] for key in
                          ("qpos_delta", "image_mean_absolute_difference", "changed_pixels",
                           "cube_translation_norm_m", "reward", "terminated", "truncated", "success")}), flush=True)
        return 0
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error),
                           "traceback": traceback.format_exc()}
        traceback.print_exc()
        return 1
    finally:
        write_json(output / "environment.json", report)
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
