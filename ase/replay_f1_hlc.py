"""Replay a frozen-LLC F1 HLC with speed commands in Isaac Gym or MuJoCo."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def export_numpy_replay(checkpoint, llc_checkpoint, policy_npz, llc_policy_npz):
    """Export the two deterministic actors on a host with working PyTorch."""
    from replay_f1 import export_actor_npz
    from utils.f1_hlc import FrozenF1LLC, load_hlc_actor
    llc = FrozenF1LLC(llc_checkpoint, "cpu")
    _, saved = load_hlc_actor(checkpoint, llc, "cpu")
    metadata = {key: saved[key] for key in
                ("epoch", "training_metadata", "hlc_config", "llc_dependency", "control_config")}
    weights = {key[len("a2c_network."):]: value.detach().float().numpy()
               for key, value in saved["model"].items()
               if key.startswith(("a2c_network.actor_mlp.", "a2c_network.mu."))}
    weights.update(mean=saved["running_mean_std"]["running_mean"].float().numpy(),
                   scale=np.sqrt(saved["running_mean_std"]["running_var"].float().numpy() + 1e-5),
                   metadata=np.array(json.dumps(metadata)),
                   source_sha256=np.array(hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()))
    np.savez_compressed(policy_npz, **weights)
    export_actor_npz(llc_checkpoint, llc_policy_npz)


def load_numpy_replay(checkpoint, llc_checkpoint, policy_npz, llc_policy_npz):
    """Load source-bound NumPy actors without importing PyTorch."""
    from replay_f1 import load_actor_numpy
    with np.load(policy_npz, allow_pickle=False) as archive:
        weights = {key: archive[key] for key in archive.files}
    saved = json.loads(str(weights["metadata"]))
    if hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() != str(weights["source_sha256"]):
        raise ValueError("NumPy HLC weights belong to a different checkpoint")
    llc = load_actor_numpy(llc_policy_npz)
    expected = saved["llc_dependency"]["sha256"]
    if llc.source_sha256 != expected or hashlib.sha256(Path(llc_checkpoint).read_bytes()).hexdigest() != expected:
        raise ValueError("NumPy HLC checkpoint requires a different LLC")
    llc.metadata = saved["llc_dependency"]
    layers = sorted(int(key.split(".")[1]) for key in weights
                    if key.startswith("actor_mlp.") and key.endswith(".weight"))

    def policy(obs):
        hidden = np.clip((np.asarray(obs, dtype=np.float32) - weights["mean"]) / weights["scale"], -5, 5)
        for index in layers:
            name = "actor_mlp." + str(index)
            hidden = np.maximum(hidden @ weights[name + ".weight"].T + weights[name + ".bias"], 0)
        return np.tanh(hidden @ weights["mu.weight"].T + weights["mu.bias"])

    return llc, policy, saved


def numpy_unit_latent(value):
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    fallback = np.zeros_like(value)
    fallback[..., 0] = 1
    return np.where(norm > 1e-8, value / np.maximum(norm, 1e-8), fallback)


def numpy_task_reward(velocity, facing, target, foot_speed_sq, leg_velocity_sq, config):
    cost = (config["speedTrackingScale"] * (velocity[0] - target) ** 2
            + config["lateralTrackingScale"] * velocity[1] ** 2
            + config["headingTrackingScale"] * (1 - facing))
    if target <= 1e-6:
        cost += (config.get("standFootSpeedScale", 0.) * foot_speed_sq
                 + config.get("standLegVelocityScale", 0.) * leg_velocity_sq)
    return float(np.exp(-cost))


def replay_summary(path, config):
    """Summarize command segments, including sustained quiet after a zero command."""
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    segments = []
    start = 0
    dt = float(rows[0]["time_s"])
    for end in range(1, len(rows) + 1):
        if end < len(rows) and rows[end]["target_speed"] == rows[start]["target_speed"]:
            continue
        block = rows[start:end]
        target = float(block[0]["target_speed"])
        begin_time = float(block[0]["time_s"]) - dt
        stable = [r for r in block if float(r["time_s"]) > begin_time + 2]
        item = dict(target=target, seconds=len(block) * dt,
                    steady_vx=float(np.mean([float(r["world_vx"]) for r in stable])) if stable else None,
                    steady_mae=float(np.mean([abs(float(r["world_vx"]) - target) for r in stable])) if stable else None)
        if target <= 1e-6:
            quiet_steps, settle = 0, None
            for row in block:
                quiet = (math.hypot(float(row["world_vx"]), float(row["world_vy"])) < config.get("quietPlanarSpeed", .05)
                         and float(row["foot_speed_rms_m_s"]) < config.get("quietFootSpeed", .1)
                         and float(row["leg_velocity_rms_rad_s"]) < config.get("quietLegVelocity", .3)
                         and not any(reason in row["termination"] for reason in ("base_height", "base_tilt", "illegal_ground_contact")))
                quiet_steps = quiet_steps + 1 if quiet else 0
                if settle is None and quiet_steps * dt >= .5 - 1e-8:
                    settle = float(row["time_s"]) - begin_time
            tail = [r for r in block if float(r["time_s"]) > float(block[-1]["time_s"]) - 5]
            item.update(settle_s=settle, final_quiet_s=quiet_steps * dt,
                        final5_path_m=sum(math.hypot(float(r["world_vx"]), float(r["world_vy"])) * dt for r in tail))
        segments.append(item)
        start = end
    print("HLC_REPLAY_SEGMENTS " + json.dumps(segments), flush=True)


def replay_mujoco(args, speeds):
    from replay_f1 import MujocoReplay, Video, joint_names, initial_case

    np.random.seed(args.seed)
    if args.policy_npz:
        llc, policy, saved = load_numpy_replay(
            args.checkpoint, args.llc_checkpoint, args.policy_npz, args.llc_policy_npz)
    else:
        import torch
        from utils.f1_hlc import FrozenF1LLC, load_hlc_actor, unit_latent, speed_task_reward
        torch.set_num_threads(4)
        torch.manual_seed(args.seed)
        llc = FrozenF1LLC(args.llc_checkpoint, "cpu")
        policy, saved = load_hlc_actor(args.checkpoint, llc, "cpu")
    metadata = saved["training_metadata"]
    cfg = dict(env=metadata["environment"], sim=metadata["simulation"])
    dt = cfg["sim"]["dt"] * cfg["env"]["controlFrequencyInv"]
    interval = saved["hlc_config"]["llc_steps"]
    if interval != cfg["env"]["hlc"]["llcSteps"]:
        raise ValueError("Checkpoint task and controller intervals differ")
    for key, value in llc.control_config.items():
        if cfg["env"].get(key) != value:
            raise ValueError("HLC and LLC control configuration differs: " + key)
    hold_steps = max(1, math.ceil(args.hold_seconds / dt / interval)) * interval
    cfg["env"]["episodeLength"] = len(speeds) * hold_steps + 1
    args.output.mkdir(parents=True, exist_ok=True)
    names = joint_names()
    engine = MujocoReplay(cfg, names, record_video=not args.no_video)
    feet = [engine.model.body(n).id for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
    leg_ids = [i for i, n in enumerate(names) if any(part in n for part in ("hip", "knee", "ankle"))]
    video = None
    physical_step, hlc_step = 0, 0
    errors, reasons = [], []
    try:
        if not math.isclose(engine.dt, dt, rel_tol=1e-6):
            raise ValueError("MuJoCo and checkpoint control timesteps differ")
        engine.reset(initial_case(0, cfg, names))
        video = Video(args.output / "mujoco_hlc.mp4") if not args.no_video else None
        with (args.output / "mujoco_hlc.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["hlc_step", "llc_step", "time_s", "target_speed", "world_vx", "world_vy",
                "base_x", "base_y", "base_z", "body_vx", "body_vy", "body_yaw_rate", "tilt_rad",
                "task_reward", "termination", "foot_speed_rms_m_s", "leg_velocity_rms_rad_s",
                *["z_" + str(i) for i in range(32)],
                *["q_" + n for n in names]])
            while physical_step < len(speeds) * hold_steps:
                target = speeds[min(physical_step // hold_steps, len(speeds) - 1)]
                rot = engine.data.xmat[engine.base].reshape(3, 3)
                direction = rot.T @ np.array([1., 0., 0.])
                command = np.array([direction[0], direction[1], target], dtype=np.float32)
                obs = np.concatenate((engine.obs()[0], command))[None, :]
                z = (numpy_unit_latent(policy(obs)) if args.policy_npz else
                     unit_latent(policy(torch.from_numpy(obs))))
                hlc_step += 1
                for _ in range(interval):
                    action = llc(engine.obs() if args.policy_npz else torch.from_numpy(engine.obs()), z)
                    _, values, reasons, q = engine.step(action)
                    physical_step += 1
                    velocity = np.zeros(6)
                    engine.mj.mj_objectVelocity(engine.model, engine.data,
                        engine.mj.mjtObj.mjOBJ_BODY, engine.base, velocity, 0)
                    vx, vy = velocity[3:5]
                    facing = engine.data.xmat[engine.base].reshape(3, 3)[0, 0]
                    hlc = cfg["env"]["hlc"]
                    foot_velocities = []
                    for body_id in feet:
                        foot_velocity = np.zeros(6)
                        engine.mj.mj_objectVelocity(engine.model, engine.data,
                            engine.mj.mjtObj.mjOBJ_BODY, body_id, foot_velocity, 0)
                        foot_velocities.append(foot_velocity[3:])
                    foot_speed_sq = float(np.square(foot_velocities).sum(-1).mean())
                    leg_velocity_sq = float(np.square(engine.data.qvel[engine.vids[leg_ids]]).mean())
                    if args.policy_npz:
                        reward = numpy_task_reward(velocity[3:], facing, target, foot_speed_sq, leg_velocity_sq, hlc)
                    else:
                        reward = float(speed_task_reward(
                            torch.tensor(velocity[None, 3:]), torch.tensor([facing]), torch.tensor([target]),
                            torch.tensor([foot_speed_sq]), torch.tensor([leg_velocity_sq]), hlc)[0])
                    errors.append(abs(vx - target))
                    writer.writerow([hlc_step, physical_step, physical_step * dt, target, vx, vy,
                        *values, reward, "+".join(reasons), math.sqrt(foot_speed_sq),
                        math.sqrt(leg_velocity_sq), *z[0].tolist(), *q])
                    if video and ((physical_step - 1) % 2 == 0 or reasons):
                        label = ("F1 HLC epoch{} | MuJoCo | frozen LLC epoch{} | seed{}\n".format(
                            saved["epoch"], llc.metadata["epoch"], args.seed)
                            + "target {:.2f}m/s | vx {:.2f}m/s | {:.2f}s | ".format(
                                target, vx, physical_step * dt)
                            + ("+".join(reasons) if reasons else "policy rollout"))
                        video.write(engine.render(), label)
                    if reasons:
                        break
                if reasons:
                    break
        print(dict(engine="mujoco", hlc_epoch=saved["epoch"], llc_epoch=llc.metadata["epoch"],
                   seed=args.seed, physical_steps=physical_step, hlc_decisions=hlc_step,
                   seconds=physical_step * dt, speed_mae=float(np.mean(errors)),
                   termination=reasons), flush=True)
        replay_summary(args.output / "mujoco_hlc.csv", cfg["env"]["hlc"])
    finally:
        try:
            if video:
                video.close()
        finally:
            engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("gym", "mujoco"), default="gym")
    parser.add_argument("--checkpoint", type=Path, required=True, help="HLC checkpoint")
    parser.add_argument("--llc-checkpoint", type=Path, required=True)
    parser.add_argument("--policy-npz", type=Path, help="Exported HLC actor for MuJoCo without PyTorch")
    parser.add_argument("--llc-policy-npz", type=Path, help="Exported matching LLC actor; requires --policy-npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", choices=("tracking", "stop-start"), default="tracking")
    parser.add_argument("--speeds", help="Explicit speeds override --protocol")
    parser.add_argument("--hold-seconds", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    if bool(args.policy_npz) != bool(args.llc_policy_npz):
        parser.error("--policy-npz and --llc-policy-npz must be supplied together")
    if args.policy_npz and args.engine != "mujoco":
        parser.error("NumPy actors are supported with mujoco only")
    default_speeds = "0,0.5,1,1.5,2,1,0,0.5,0" if args.protocol == "stop-start" else "1.0,0.6,1.6,1.0"
    speeds = [float(s) for s in (args.speeds or default_speeds).split(",")]
    if not speeds or not np.isfinite(speeds).all() or min(speeds) < 0 or not np.isfinite(args.hold_seconds) or args.hold_seconds <= 0:
        parser.error("speeds must be finite/nonnegative and hold-seconds positive")
    if args.engine == "mujoco":
        return replay_mujoco(args, speeds)
    import isaacgym
    import torch
    from replay_f1 import GymReplay, Video, joint_names, initial_case, kinematics
    from utils.f1_hlc import FrozenF1LLC, control_interval, load_hlc_actor
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    llc = FrozenF1LLC(args.llc_checkpoint, "cuda:0")
    policy, saved = load_hlc_actor(args.checkpoint, llc, "cuda:0")
    metadata = saved["training_metadata"]
    cfg = dict(env=metadata["environment"], sim=metadata["simulation"])
    dt = cfg["sim"]["dt"]*cfg["env"]["controlFrequencyInv"]
    interval = saved["hlc_config"]["llc_steps"]
    if interval != cfg["env"]["hlc"]["llcSteps"]:
        raise ValueError("Checkpoint task and controller intervals differ")
    hold_steps = max(1, math.ceil(args.hold_seconds/dt/interval))*interval
    cfg["env"]["episodeLength"] = len(speeds)*hold_steps+1
    args.output.mkdir(parents=True, exist_ok=True)
    names = joint_names()
    engine = GymReplay(cfg, names, record_video=not args.no_video)
    video = None
    try:
        engine.task.validate_control_config(llc.control_config)
        engine.reset(initial_case(0, cfg, names))
        engine.task.command_next_step[:] = cfg["env"]["episodeLength"]+1
        video = Video(args.output / "isaac_hlc.mp4") if not args.no_video else None
        physical_step = 0
        hlc_step = 0
        errors = []
        with (args.output / "isaac_hlc.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["hlc_step", "llc_step", "time_s", "target_speed", "world_vx", "world_vy",
                "base_x", "base_y", "base_z", "body_vx", "body_vy", "body_yaw_rate", "tilt_rad",
                "task_reward", "termination", "foot_speed_rms_m_s", "leg_velocity_rms_rad_s",
                *["z_"+str(i) for i in range(32)], *["q_"+n for n in names]])

            def capture(task, z, action):
                nonlocal physical_step
                physical_step += 1
                root = task._humanoid_root_states[0].cpu().numpy()
                values = kinematics(root[:3], root[[6, 3, 4, 5]], root[7:10], root[10:13])
                why = [k for k, mask in task.extras["termination_reasons"].items() if bool(mask[0])]
                target = float(task.target_speed[0])
                foot_speed_sq, leg_velocity_sq = task._standing_motion()
                errors.append(abs(root[7]-target))
                writer.writerow([hlc_step, physical_step, physical_step*dt, target, root[7], root[8],
                    *values, float(task.rew_buf[0]), "+".join(why),
                    float(foot_speed_sq[0].sqrt()), float(leg_velocity_sq[0].sqrt()), *z[0].cpu().tolist(),
                    *task._dof_pos[0].cpu().tolist()])
                if video and (physical_step-1) % 2 == 0:
                    label = ("F1 HLC epoch{} | frozen ASE LLC epoch{} | seed{}\n".format(saved["epoch"], llc.metadata["epoch"], args.seed)
                             + "target {:.2f}m/s | vx {:.2f}m/s | {:.2f}s | ".format(target, root[7], physical_step*dt)
                             + ("+".join(why) if why else "policy rollout"))
                    video.write(engine.render(), label)

            while physical_step < len(speeds)*hold_steps:
                index = min(physical_step//hold_steps, len(speeds)-1)
                engine.task.target_speed[:] = speeds[index]
                engine.task._compute_observations()
                hlc_step += 1
                _, _, done, info = control_interval(engine.task, llc, policy(engine.task.obs_buf), interval, on_step=capture)
                if bool(done[0]):
                    break
        print(dict(hlc_epoch=saved["epoch"], llc_epoch=llc.metadata["epoch"], seed=args.seed,
                   physical_steps=physical_step, hlc_decisions=hlc_step, seconds=physical_step*dt,
                   speed_mae=float(np.mean(errors)), termination=[k for k, v in info["termination_reasons"].items() if bool(v[0])]), flush=True)
        replay_summary(args.output / "isaac_hlc.csv", cfg["env"]["hlc"])
    finally:
        try:
            if video:
                video.close()
        finally:
            engine.close()


if __name__ == "__main__":
    main()
