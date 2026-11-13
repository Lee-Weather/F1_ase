"""Replay a frozen-LLC F1 HLC with explicit speed commands in Isaac Gym."""
import argparse
import csv
import math
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="HLC checkpoint")
    parser.add_argument("--llc-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speeds", default="1.0,0.8,1.2,1.0")
    parser.add_argument("--hold-seconds", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    speeds = [float(s) for s in args.speeds.split(",")]
    if not speeds or not np.isfinite(speeds).all() or min(speeds) < 0 or not np.isfinite(args.hold_seconds) or args.hold_seconds <= 0:
        parser.error("speeds must be finite/nonnegative and hold-seconds positive")
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
                "task_reward", "termination", *["z_"+str(i) for i in range(32)], *["q_"+n for n in names]])

            def capture(task, z, action):
                nonlocal physical_step
                physical_step += 1
                root = task._humanoid_root_states[0].cpu().numpy()
                values = kinematics(root[:3], root[[6, 3, 4, 5]], root[7:10], root[10:13])
                why = [k for k, mask in task.extras["termination_reasons"].items() if bool(mask[0])]
                target = float(task.target_speed[0])
                errors.append(abs(root[7]-target))
                writer.writerow([hlc_step, physical_step, physical_step*dt, target, root[7], root[8],
                    *values, float(task.rew_buf[0]), "+".join(why), *z[0].cpu().tolist(),
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
    finally:
        try:
            if video:
                video.close()
        finally:
            engine.close()


if __name__ == "__main__":
    main()
