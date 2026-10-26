"""Capture at most 20 PushCube control steps and encode only the saved PNGs.

No training, extra warm-up steps, automatic resets or successful-run retries.
CPU physics; CPU-accessible SAPIEN/Vulkan RGB. Existing results are preserved.
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


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "result")
    parser.add_argument("--reference-env", type=Path, help="Optional article-3 raw or public environment JSON")
    parser.add_argument("--seed", type=int, required=True, choices=(43, 44))
    parser.add_argument("--budget-ledger", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20, choices=range(1, 21))
    args = parser.parse_args()
    from capture_protocol import reserve_step

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("frames", "trajectory.npz", "trajectory.json", "environment.json", "replay.gif", "schema.json")):
        parser.error("Results exist; choose a fresh output directory. Never overwrite a capture.")
    frames_dir = output / "frames"
    frames_dir.mkdir()
    cache = Path(os.environ.get("SIM_CAMERA_CACHE", str(output / ".cache"))).resolve()
    for key, folder in (("MS_ASSET_DIR", "maniskill-assets"), ("HF_HOME", "huggingface"),
                        ("TORCH_HOME", "torch"), ("MPLCONFIGDIR", "matplotlib")):
        path = cache / folder
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(key, str(path))
    os.environ.setdefault("MS_SKIP_ASSET_DOWNLOAD_PROMPT", "1")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    report = {"status": "started", "captured_at_utc": datetime.now(timezone.utc).isoformat(),
              "python_version": platform.python_version(),
              "packages": {n: metadata.version(n) for n in ("mani-skill", "sapien", "torch", "gymnasium", "numpy", "pillow")},
              "experiment": {"task": "PushCube-v1", "robot": "panda", "num_envs": 1, "seed": args.seed,
                             "obs_mode": "rgb", "control_mode": "pd_joint_delta_pos", "camera": "base_camera",
                             "camera_role": "policy_observation_sensor", "width": 256, "height": 256,
                             "shader_pack": "minimal", "sim_backend": "cpu", "render_backend": "cpu",
                             "requested_steps": args.steps, "max_authorized_step_calls": 20,
                             "explicit_reset_calls": 0, "attempted_step_calls": 0, "completed_steps": 0,
                             "training": False},
              "local_paths": {"python_executable": sys.executable, "cache": str(cache),
                              "reference_env": str(args.reference_env.resolve()) if args.reference_env else None},
              "script_sha256": sha(__file__)}
    env = None
    try:
        import gymnasium as gym
        import mani_skill.envs
        import numpy as np
        import torch
        from PIL import Image

        def array(value):
            return value.detach().cpu().numpy().copy() if isinstance(value, torch.Tensor) else np.array(value, copy=True)

        def flatten(value, prefix=""):
            if isinstance(value, dict):
                result = {}
                for key, item in value.items():
                    result.update(flatten(item, prefix+"/"+key if prefix else key))
                return result
            item = array(value)
            if item.dtype.hasobject or not np.isfinite(item).all():
                raise ValueError("Invalid numeric field: "+prefix)
            return {prefix: item}

        def fingerprint(observation):
            h = hashlib.sha256()
            for key, value in sorted(observation.items()):
                h.update(key.encode()); h.update(value.tobytes())
            return h.hexdigest()

        def save_frame(observation, index):
            rgb = observation["sensor_data/base_camera/rgb"]
            assert rgb.shape == (1, 256, 256, 3) and rgb.dtype == np.uint8
            path = frames_dir / f"{index:04d}.png"
            Image.fromarray(rgb[0]).save(path)
            return {"index": index, "file": "frames/"+path.name, "sha256": sha(path)}

        env = gym.make("PushCube-v1", robot_uids="panda", num_envs=1, obs_mode="rgb",
                       sim_backend="cpu", render_backend="cpu", control_mode="pd_joint_delta_pos",
                       render_mode="rgb_array", sensor_configs={"width": 256, "height": 256, "shader_pack": "minimal"})
        obs, reset_info = env.reset(seed=args.seed)
        report["experiment"]["explicit_reset_calls"] = 1
        base, combined = env.unwrapped, env.unwrapped.agent.controller
        report["resolved_backend"] = {key: str(getattr(base.backend, key)) for key in
                                      ("sim_backend", "render_backend", "device", "sim_device", "render_device")}
        report["timing"] = {"control_freq_hz": base.control_freq, "simulation_freq_hz": base.sim_freq,
                            "control_dt_s": base.control_timestep, "simulation_dt_s": base.sim_timestep,
                            "physics_substeps_per_control": base._sim_steps_per_control}
        report["action_space"] = {"shape": list(env.action_space.shape), "dtype": str(env.action_space.dtype),
                                  "low": env.action_space.low.tolist(), "high": env.action_space.high.tolist()}
        report["controllers"] = {name: {"class": type(part).__name__, "config": asdict(part.config),
                                        "joint_names": [j.name for j in part.joints],
                                        "active_joint_indices": array(part.active_joint_indices).tolist(),
                                        "physical_low": part._original_single_action_space.low.tolist(),
                                        "physical_high": part._original_single_action_space.high.tolist(),
                                        "action_slice": list(combined.action_mapping[name])}
                                 for name, part in combined.controllers.items()}
        pkg = Path(metadata.distribution("mani-skill").locate_file("mani_skill"))
        report["source_sha256"] = {name: sha(pkg / name) for name in
                                    ("agents/robots/panda/panda.py", "agents/controllers/base_controller.py",
                                     "agents/controllers/pd_joint_pos.py", "utils/gym_utils.py",
                                     "envs/tasks/tabletop/push_cube.py", "envs/sapien_env.py")}
        if args.reference_env:
            ref = json.loads(args.reference_env.read_text(encoding="utf-8"))
            for key in ("packages", "controllers", "timing", "action_space", "resolved_backend", "source_sha256"):
                assert report[key] == ref[key], "Article-3 runtime/source mismatch: "+key
            report["article3_reference_match"] = True
        # Read actual API device information in this formal environment only.
        import sapien
        render_device = base.scene.sub_scenes[0].get_render_system().device
        report["render_provenance"] = {
            "physics_backend": report["resolved_backend"]["sim_backend"],
            "render_backend": report["resolved_backend"]["render_backend"],
            "render_system_device": str(render_device),
            "render_system_device_name": render_device.name,
            "device_is_cpu": render_device.is_cpu(),
            "device_is_cuda": render_device.is_cuda(),
            "device_can_render": render_device.can_render(),
            "device_pci_string": render_device.pci_string,
            "available_device_summary": sapien.render.get_device_summary(),
            "image_transfer": "CPU-accessible RGB copied into numpy uint8; no CUDA image tensors",
            "physical_render_device_note": "Device API and enumeration are recorded separately. CPU-accessible memory does not establish CPU-only rasterization. No GPU utilization benchmark performed."}
        arm, gripper = combined.controllers["arm"], combined.controllers["gripper"]
        assert combined.action_mapping == {"arm": (0, 7), "gripper": (7, 8)}
        assert arm.config.normalize_action and arm.config.use_delta and not arm.config.use_target
        assert gripper.config.normalize_action and not gripper.config.use_delta
        assert np.allclose(arm._original_single_action_space.low, -0.1) and np.allclose(arm._original_single_action_space.high, 0.1)
        assert np.allclose(gripper._original_single_action_space.low, -0.01) and np.allclose(gripper._original_single_action_space.high, 0.04)
        plan = np.zeros((20, 8), dtype=np.float32)
        plan[:, 7] = 1
        plan[:, 0] = np.repeat(np.array([0.1, -0.1, 0.1, -0.1], np.float32), 5)
        assert all(env.action_space.contains(a) for a in plan)

        observations, infos, actions, rewards, terms, truncs, targets = [flatten(obs)], [], [], [], [], [], []
        cubes, indices = [array(base.obj.pose.raw_pose)], [array(base.elapsed_steps)]
        hashes = [fingerprint(observations[0])]
        frame_records = [save_frame(observations[0], 0)]
        stop_reason = "requested_step_limit"
        for k in range(args.steps):
            assert report["experiment"]["attempted_step_calls"] < 20
            action = plan[k].copy()
            report["experiment"]["attempted_step_calls"] += 1
            reserve_step(args.budget_ledger, args.seed, k + 1)
            write_json(output / "environment.json", report)
            next_obs, reward, terminated, truncated, info = env.step(action)
            # No reset after this call, including a terminal transition.
            actions.append(action.copy()); rewards.append(array(reward))
            terms.append(array(terminated)); truncs.append(array(truncated)); infos.append(flatten(info))
            observations.append(flatten(next_obs)); cubes.append(array(base.obj.pose.raw_pose))
            indices.append(array(base.elapsed_steps))
            targets.append({name: array(part._target_qpos) for name, part in combined.controllers.items()})
            hashes.append(fingerprint(observations[-1]))
            frame_records.append(save_frame(observations[-1], k+1))
            report["experiment"]["completed_steps"] = len(actions)
            print(json.dumps({"step": k+1, "joint1_input": float(action[0]), "terminated": terms[-1].tolist(),
                              "truncated": truncs[-1].tolist()}), flush=True)
            if terms[-1].any() or truncs[-1].any():
                stop_reason = "terminated_or_truncated"
                break
        n = len(actions)
        assert 1 <= n <= 20 and all(fingerprint(o) == h for o, h in zip(observations, hashes))
        assert all(set(o) == set(observations[0]) for o in observations)
        arrays = {"obs/"+key: np.stack([o[key] for o in observations]) for key in observations[0]}
        arrays.update({"info/"+key: np.stack([i[key] for i in infos]) for key in infos[0]})
        arrays.update({"reset_info/"+key: value for key, value in flatten(reset_info).items()})
        arrays.update(actions=np.stack(actions), rewards=np.stack(rewards), terminated=np.stack(terms),
                      truncated=np.stack(truncs), step_index=np.stack(indices), **{
                          "diagnostic/cube_pose": np.stack(cubes),
                          "diagnostic/arm_target_qpos": np.stack([t["arm"] for t in targets]),
                          "diagnostic/gripper_target_qpos": np.stack([t["gripper"] for t in targets])})
        arrays["time_s"] = arrays["step_index"].astype(np.float64)*base.control_timestep
        np.savez_compressed(output / "trajectory.npz", **arrays)

        # Encode saved original PNGs only. No additional simulation or altered raw data.
        palette_frames = []
        for record in frame_records:
            with Image.open(output / record["file"]) as original:
                palette_frames.append(original.convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
        frame_ms = round(1000*base.control_timestep)
        assert frame_ms > 0 and frame_ms % 10 == 0
        palette_frames[0].save(output / "replay.gif", save_all=True, append_images=palette_frames[1:],
                               duration=[frame_ms]*len(palette_frames), loop=0, optimize=False, disposal=2)
        with Image.open(output / "replay.gif") as replay:
            durations = []
            for i in range(replay.n_frames):
                replay.seek(i); replay.load(); durations.append(int(replay.info.get("duration", 0)))
            replay_meta = {"frames": replay.n_frames, "durations_ms": durations,
                           "total_duration_ms": sum(durations), "loop": replay.info.get("loop"),
                           "width": replay.width, "height": replay.height,
                           "source_pngs": len(frame_records), "nominal_frame_duration_ms": frame_ms,
                           "nominal_fps": 1000/frame_ms, "merged_frames": len(frame_records)-replay.n_frames,
                           "extra_final_state_hold_ms": frame_ms,
                           "note": "Presentation only: per-frame 256-color quantization. Source PNGs and NPZ RGB are the original data. Last state is held one frame interval after simulated end; looping restarts the player, not the simulator."}
        q = arrays["obs/agent/qpos"]
        rgb = arrays["obs/sensor_data/base_camera/rgb"]
        cube_delta = arrays["diagnostic/cube_pose"][-1, :, :3]-arrays["diagnostic/cube_pose"][0, :, :3]
        per_step = []
        for k in range(n):
            diff = np.abs(rgb[k+1].astype(np.int16)-rgb[k].astype(np.int16))
            per_step.append({"action_index": k, "obs_index": k, "next_obs_index": k+1,
                             "time_t_s": float(arrays["time_s"][k, 0]), "time_tp1_s": float(arrays["time_s"][k+1, 0]),
                             "action": actions[k].tolist(), "joint1_target_rad": float(targets[k]["arm"][0, 0]),
                             "joint1_actual_delta_rad": float(q[k+1, 0, 0]-q[k, 0, 0]),
                             "changed_pixels": int(np.any(diff != 0, axis=-1).sum()),
                             "reward": rewards[k].tolist(), "terminated": terms[k].tolist(),
                             "truncated": truncs[k].tolist(), "info": {key: value.tolist() for key, value in infos[k].items()}})
        trajectory = {"format": "time_stacked_numeric_arrays_v1", "N": n, "observations": n+1,
                      "stop_reason": stop_reason, "planned_action_segments": [{"steps_inclusive": [1+5*i, 5+5*i], "joint1_normalized_input": value} for i, value in enumerate([0.1, -0.1, 0.1, -0.1])],
                      "simulation_duration_s": float(arrays["time_s"][-1, 0]-arrays["time_s"][0, 0]),
                      "states": [{**r, "time_s": float(arrays["time_s"][i, 0])} for i, r in enumerate(frame_records)],
                      "transitions": per_step, "snapshot_fingerprints": hashes,
                      "independent_snapshots_unchanged": True,
                      "statistics": {"joint1_start_rad": float(q[0, 0, 0]), "joint1_end_rad": float(q[-1, 0, 0]),
                                     "joint1_min_rad": float(q[:, 0, 0].min()), "joint1_max_rad": float(q[:, 0, 0].max()),
                                     "joint1_net_delta_rad": float(q[-1, 0, 0]-q[0, 0, 0]),
                                     "cube_net_translation_m": cube_delta.tolist(), "cube_net_translation_norm_m": float(np.linalg.norm(cube_delta)),
                                     "success_any": bool(arrays["info/success"].any()), "success_final": arrays["info/success"][-1].tolist(),
                                     "first_png_sha256": frame_records[0]["sha256"], "last_png_sha256": frame_records[-1]["sha256"]},
                      "replay": replay_meta}
        schema = {"note": "For k in [0,N): obs[k] -> actions[k] -> obs[k+1]. Do not concatenate across a terminal reset.",
                  "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                                   "time_axis": None if key.startswith("reset_info/") else 0,
                                   "environment_axis": None if key == "actions" or key.startswith("reset_info/") else 1}
                             for key, value in arrays.items()},
                  "units": {"obs/agent/qpos": ["rad"]*7+["m"]*2, "obs/agent/qvel": ["rad/s"]*7+["m/s"]*2,
                            "obs/extra/tcp_pose": "world x,y,z in m; quaternion qw,qx,qy,qz dimensionless",
                            "obs/sensor_data/base_camera/rgb": "uint8 RGB 0..255; axes time,environment,height,width,channel",
                            "obs/sensor_param/base_camera/extrinsic_cv": "world-to-camera [R|t], OpenCV axes; translation m",
                            "obs/sensor_param/base_camera/cam2world_gl": "camera-to-world matrix, OpenGL axes; translation m",
                            "obs/sensor_param/base_camera/intrinsic_cv": "OpenCV intrinsics; focal/principal point in pixels",
                            "actions": "normalized dimensionless; first7 delta_rad=0.1*a, last absolute finger target_m=0.015+0.025*a",
                            "diagnostic/cube_pose": "simulator-only world xyz(m),qw,qx,qy,qz; excluded from policy obs",
                            "diagnostic/arm_target_qpos": "commanded angles in rad; separate from measured positions",
                            "diagnostic/gripper_target_qpos": "commanded positions of two fingers, m",
                            "time_s": "simulation seconds relative to seeded reset", "step_index": "control-step index",
                            "rewards": "task reward, not a success flag", "terminated": "task terminal-state flag",
                            "truncated": "external/time-limit cutoff flag", "info/*": "per-action returned diagnostics",
                            "reset_info/*": "reset-only diagnostics, not a time series"},
                  "joint_order": [j.name for j in base.agent.robot.get_active_joints()]}
        write_json(output / "trajectory.json", trajectory)
        write_json(output / "schema.json", schema)
        report.update(status="capture_passed_pending_independent_validation", replay=replay_meta,
                      torch_cuda_build=torch.version.cuda,
                      artifacts_sha256={name: sha(output / name) for name in ("trajectory.npz", "trajectory.json", "schema.json", "replay.gif")})
        print(json.dumps({"N": n, "observations": n+1, "arrays": len(arrays), "stop_reason": stop_reason,
                          "simulation_duration_s": trajectory["simulation_duration_s"], "replay": replay_meta,
                          "statistics": trajectory["statistics"]}), flush=True)
        return 0
    except Exception as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        traceback.print_exc()
        return 1
    finally:
        write_json(output / "environment.json", report)
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
