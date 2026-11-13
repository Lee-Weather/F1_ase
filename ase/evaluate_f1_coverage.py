"""Diagnose F1 ASE reference coverage; this never updates the policy.

Random fixed-z rollouts measure accessible behavior. Reference-encoder sequences
from default and matched states are additional retrieval probes, not proof that
an unobserved skill is absent. Videos contain closed-loop policy output only.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from replay_f1 import ROOT, GymReplay, Latents, Video, initial_case, joint_names, kinematics, load_actor


def random_rollouts(engine, actor, count, seed):
    """Independent fixed-z probes in parallel simulator environments."""
    import torch
    task = engine.task
    task.reset()
    roots = task._humanoid_root_states.cpu().numpy().copy()
    states = [dict(root=np.r_[r[:3], r[[6, 3, 4, 5]]], init="default") for r in roots]
    z = np.stack([Latents(seed+i, False).next() for i in range(count)])
    latent = torch.as_tensor(z, device=engine.device)
    features = [[f.copy()] for f in task._curr_amp_obs_buf.cpu().numpy()]
    rows = [[] for _ in range(count)]
    active = np.ones(count, dtype=bool)
    for step in range(499):
        action = actor(task.obs_buf, latent)
        task.step(action)
        engine.gym.fetch_results(engine.sim, True)
        roots = task._humanoid_root_states.cpu().numpy()
        joints = task._dof_pos.cpu().numpy()
        amp = task._curr_amp_obs_buf.cpu().numpy()
        reasons = {k: v.cpu().numpy() for k, v in task.extras["termination_reasons"].items()}
        for i in np.flatnonzero(active):
            r = roots[i]
            values = kinematics(r[:3], r[[6, 3, 4, 5]], r[7:10], r[10:13])
            why = [k for k, v in reasons.items() if v[i]]
            if step == 498 and not why:
                why = ["probe_end"]
            rows[i].append((values, why, joints[i].copy()))
            features[i].append(amp[i].copy())
            active[i] = not bool(why)
        if not active.any():
            break
    print("RANDOM_BATCH_DONE " + str(count), flush=True)
    return [(states[i], np.asarray(features[i]), np.repeat(z[i:i+1], len(rows[i]), 0), rows[i])
            for i in range(count)]


def analyze_saved(output, device="cuda:0", prefix=""):
    """Add moving-phase and time-aligned diagnostics without rerunning physics."""
    import torch
    summary_path = output / (prefix + "coverage_summary.json")
    report = json.loads(summary_path.read_text())
    with np.load(output / (prefix + "coverage_features.npz")) as archive:
        arrays = {k: archive[k] for k in archive.files}
    current = torch.as_tensor(arrays["reference"], device=device)
    counts = arrays["reference_counts"]
    offsets = np.r_[0, np.cumsum(counts)]
    chunks = [current[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
    mean = torch.stack([x.mean(0) for x in chunks]).mean(0)
    second = torch.stack([x.square().mean(0) for x in chunks]).mean(0)
    std = (second - mean.square()).clamp(min=0).sqrt()
    scale = std.clone()
    for start, end, floor in [(0, 1, .1), (1, 7, .1), (7, 10, .5), (10, 13, .5),
                              (13, 42, .1), (42, 71, .5), (71, 89, .1)]:
        scale[start:end] = std[start:end].clamp(min=floor) * np.sqrt(7*(end-start))
    scale = scale.repeat(2) * np.sqrt(2)
    reference = torch.as_tensor(arrays["reference_pair"], device=device) / scale
    reference_velocity = current[:, 7:9]
    reference_speed = reference_velocity.norm(dim=1)
    moving = [x[:, 7:9].norm(dim=1).cpu().numpy() > .2 for x in chunks]
    minimum = {mode: np.full(len(current), np.inf, dtype=np.float32)
               for mode in ("random_fixed", "encoded_default", "encoded_reference")}
    for case in report["cases"]:
        features = torch.as_tensor(arrays["amp"][arrays["case"] == case["case"]], device=device)
        # Pair index 24 corresponds to control step 26, after the .5s exclusion.
        pair = torch.cat((features[1:], features[:-1]), 1)[24:] / scale
        if not len(pair):
            continue
        query_velocity = features[1:][24:, 7:9]
        query_speed = query_velocity.norm(dim=1)
        cosine = (query_velocity @ reference_velocity.T) / (
            query_speed[:, None] * reference_speed[None, :]).clamp(min=1e-6)
        moving_gate = ((cosine >= .5) & (query_speed[:, None] >= .5*reference_speed[None, :])
                       & (query_speed[:, None] <= 1.5*reference_speed[None, :] + .3))
        gate = torch.where(reference_speed[None, :] > .2, moving_gate,
                           query_speed[:, None] <= .3)
        distance_matrix = torch.cdist(pair, reference).masked_fill(~gate, float("inf"))
        distance = distance_matrix.min(0).values.cpu().numpy()
        minimum[case["mode"]] = np.minimum(minimum[case["mode"]], distance)
        if case["target"]:
            j = next(i for i, row in enumerate(report["coverage"]) if row["clip"] == case["target"])
            a, b = offsets[j:j+2]
            close = distance[a:b] <= report["coverage"][j]["threshold"]
            support = (distance_matrix[:, a:b].min(1).values <= report["coverage"][j]["threshold"]).cpu().numpy()
            longest = streak = 0
            for matched in support:
                streak = streak+1 if matched else 0
                longest = max(longest, streak)
            case["target_gated_support_fraction"] = float(support.mean())
            case["target_longest_support_s"] = longest * report["dt"]
            case["target_reference_fraction"] = float(close.mean())
            case["target_moving_reference_fraction"] = float(close[moving[j]].mean()) if moving[j].any() else None
            # The action using the window at t advances the simulator to t+dt.
            aligned = reference[a+26:b]
            n = min(len(aligned), len(pair))
            case["aligned_distance_median"] = float((pair[:n]-aligned[:n]).norm(dim=1).median()) if n else None
    for j, row in enumerate(report["coverage"]):
        a, b = offsets[j:j+2]
        row["moving_reference_fraction"] = float(moving[j].mean())
        row["reference_mean_vx"] = float(chunks[j][:, 7].mean())
        for mode, distance in minimum.items():
            close = distance[a:b] <= row["threshold"]
            row[mode+"_gated_reference_fraction"] = float(close.mean())
            row[mode+"_moving_reference_fraction"] = float(close[moving[j]].mean()) if moving[j].any() else None
        for mode in ("encoded_default", "encoded_reference"):
            case = next((c for c in report["cases"] if c["mode"] == mode and c["target"] == row["clip"]), None)
            if case is None:
                continue
            for key in ("seconds", "distance_m", "mean_vx", "target_reference_fraction",
                        "target_moving_reference_fraction", "aligned_distance_median"):
                row[mode+"_own_"+key] = case.get(key)
            row[mode+"_own_target_longest_support_s"] = case.get("target_longest_support_s", 0.)
            row[mode+"_own_termination"] = "+".join(case["termination"])
    report["moving_phase_definition"] = "reference planar root speed >0.2m/s; empty subset is null"
    report["velocity_gate"] = "moving: velocity cosine>=0.5 and speed in [0.5*vref,1.5*vref+0.3]; stationary: speed<=0.3m/s"
    report["phase_caveat"] = "phase coverage is unordered; aligned distances are separate and do not establish transition success"
    report["analysis_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    arrays.update({"min_distance_"+mode: value for mode, value in minimum.items()})
    temporary = output / (prefix + "coverage_features.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(output / (prefix + "coverage_features.npz"))
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output / (prefix + "coverage_summary.csv")).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(
            key for row in report["coverage"] for key in row)))
        writer.writeheader()
        writer.writerows(report["coverage"])
    print("COVERAGE_ANALYSIS_DONE " + str(len(report["coverage"])), flush=True)


class ReferencePairs:
    """Nearest reference transitions in heading-invariant 89-D AMP features.

    Both states of each 20 ms transition are compared. Seven physical feature
    groups have equal total weight. Scale floors prevent almost constant axes
    from dominating. Acceptance is a diagnostic heuristic, never a skill pass.
    """
    def __init__(self, task):
        import torch
        self.task, self.lib = task, task._motion_lib
        self.names = [Path(p).stem for p in self.lib._motion_files]
        self.windows, self.times, self.pairs, self.slices = [], [], [], []
        offset = 0
        history = task.dt * (task._num_amp_obs_steps - 1)
        for i in range(self.lib.num_motions()):
            length = float(self.lib._motion_lengths[i])
            times = torch.arange(history, length + 1e-5, task.dt, device=task.device)
            ids = torch.full_like(times, i, dtype=torch.long)
            windows = task.build_amp_obs_demo(ids, times).reshape(len(times), 10, 89)
            self.windows.append(windows)
            self.times.append(times)
            self.pairs.append(windows[:, :2].reshape(len(times), -1))
            self.slices.append(slice(offset, offset + len(times)))
            offset += len(times)
        # Equal clip weighting when normalizing, independent of clip duration.
        means = torch.stack([w[:, 0].mean(0) for w in self.windows])
        seconds = torch.stack([w[:, 0].square().mean(0) for w in self.windows])
        self.mean = means.mean(0)
        std = (seconds.mean(0) - self.mean.square()).clamp(min=0).sqrt()
        groups = [(0, 1, .1), (1, 7, .1), (7, 10, .5), (10, 13, .5),
                  (13, 42, .1), (42, 71, .5), (71, 89, .1)]
        scale = std.clone()
        for start, end, floor in groups:
            scale[start:end] = std[start:end].clamp(min=floor) * np.sqrt(7 * (end-start))
        self.scale = scale.repeat(2) * np.sqrt(2)
        self.reference = torch.cat(self.pairs) / self.scale
        self.thresholds = []
        # Per-clip leave-0.1s-neighborhood-out reference self distance, p95.
        # Clamp [.25,.75] makes the heuristic explicit and bounded.
        for s in self.slices:
            x = self.reference[s]
            d = torch.cdist(x, x)
            index = torch.arange(len(x), device=task.device)
            d[(index[:, None] - index[None, :]).abs() <= 5] = float("inf")
            finite = d.min(1).values
            finite = finite[torch.isfinite(finite)]
            threshold = float(torch.quantile(finite, .95)) if len(finite) else .25
            self.thresholds.append(float(np.clip(threshold, .25, .75)))

    def analyze(self, features):
        import torch
        # features includes the reset state, so the first pair never crosses cases.
        x = torch.as_tensor(features, device=self.task.device)
        pairs = torch.cat((x[1:], x[:-1]), 1) / self.scale
        distances = torch.cdist(pairs, self.reference).cpu().numpy()
        best_each, phase_each, reverse_each = [], [], []
        for s in self.slices:
            block = distances[:, s]
            best_each.append(block.min(1))
            phase_each.append(block.argmin(1))
            reverse_each.append(block.min(0))
        return np.stack(best_each, 1), np.stack(phase_each, 1), reverse_each

    def state(self, motion_id):
        import torch
        ids = torch.tensor([motion_id], device=self.task.device)
        times = self.times[motion_id][:1]
        root, quat, q, vel, omega, qd, _ = self.lib.get_motion_state(ids, times)
        root = np.r_[root[0].cpu().numpy(), quat[0, [3, 0, 1, 2]].cpu().numpy()]
        root[:2] = 0
        return dict(q=q[0].cpu().numpy(), qd=qd[0].cpu().numpy(), root=root,
                    velocity=np.r_[vel[0].cpu().numpy(), omega[0].cpu().numpy()],
                    init=self.names[motion_id] + ":t" + str(float(times[0])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--random-cases", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--latent-seed", type=int, default=1000)
    parser.add_argument("--clips", type=int, nargs="+", help="One-based dataset clip numbers; default: all")
    parser.add_argument("--duration", type=float,
                        help="Seconds per encoded probe; hold the last reference z after the clip ends")
    parser.add_argument("--prefix", default="", help="Filename prefix to preserve existing replay artifacts")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Add moving-phase diagnostics to existing features; no simulation")
    args = parser.parse_args()
    if args.random_cases < 0:
        parser.error("random-cases must be nonnegative")
    if args.duration is not None and (not np.isfinite(args.duration) or args.duration <= 0):
        parser.error("duration must be finite and positive")
    if Path(args.prefix).name != args.prefix and args.prefix:
        parser.error("prefix must be a filename prefix, not a path")
    if args.analyze_only:
        analyze_saved(args.output, prefix=args.prefix)
        return
    import isaacgym  # Must precede torch.
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    actor = load_actor(args.checkpoint, "cuda:0")
    if actor.algorithm != "ASE" or not hasattr(actor, "encode"):
        parser.error("coverage encoder probes require an ASE checkpoint")
    cfg = yaml.safe_load((ROOT / "ase/data/cfg/humanoid_f1.yaml").read_text())
    cfg["env"].update(actor.control_config)
    # Only the diagnostic timeout changes; physical/control parameters stay intact.
    dt = cfg["sim"]["dt"] * cfg["env"]["controlFrequencyInv"]
    cfg["env"]["episodeLength"] = max(1001, int(np.ceil((args.duration or 0)/dt))+1)
    names = joint_names()
    engine = GymReplay(cfg, names, record_video=False, num_envs=max(1, args.random_cases))
    video = None
    summaries, all_features, all_z, all_ids = [], [], [], []
    try:
        refs = ReferencePairs(engine.task)
        selected = [i-1 for i in args.clips] if args.clips else list(range(len(refs.names)))
        if len(set(selected)) != len(selected) or any(i < 0 or i >= len(refs.names) for i in selected):
            parser.error("clips must be distinct numbers within the dataset")
        encoded = [actor.encode(w.reshape(len(w), -1)) for w in refs.windows]
        random = random_rollouts(engine, actor, args.random_cases, args.latent_seed) if args.random_cases else []
        engine.close()
        engine = None
        engine = GymReplay(cfg, names, record_video=not args.no_video)
        refs.task = engine.task
        probes = [("random_fixed", -1) for _ in range(args.random_cases)]
        probes += [(mode, i) for i in selected
                   for mode in ("encoded_default", "encoded_reference")]
        video = Video(args.output / (args.prefix + "isaac_coverage.mp4")) if not args.no_video else None
        video_frames = 0
        # Track how much of each clip's reference transition manifold is approached.
        reverse = {mode: [np.full(len(t), np.inf) for t in refs.times]
                   for mode in ("random_fixed", "encoded_default", "encoded_reference")}
        with (args.output / (args.prefix + "isaac_coverage.csv")).open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["case", "mode", "target", "latent_seed", "step", "time_s",
                             "base_x", "base_y", "base_z", "local_vx", "local_vy", "yaw_rate",
                             "tilt_rad", "termination", "nearest_clip", "nearest_distance",
                             "nearest_time_s", "within_threshold", "target_distance",
                             *["q_"+n for n in names]])
            for case, (mode, mid) in enumerate(probes):
                start_video = video_frames / 25 if mid >= 0 and video else None
                if mid < 0:
                    state, features, latents, rows = random[case]
                else:
                    state = refs.state(mid) if mode == "encoded_reference" else initial_case(0, cfg, names)
                    obs = engine.reset(state)
                    if mode == "encoded_reference":
                        engine.task._amp_obs_buf[0] = refs.windows[mid][0]
                    features = [engine.task._curr_amp_obs_buf[0].cpu().numpy().copy()]
                    latents, rows = [], []
                    steps = int(np.ceil(args.duration/dt)) if args.duration is not None else len(refs.times[mid])
                    for step in range(steps):
                        reference_step = min(step, len(refs.times[mid])-1)
                        z = encoded[mid][reference_step:reference_step+1]
                        action = actor(obs, z)
                        obs, values, reasons, q = engine.step(action)
                        if step == steps - 1 and not reasons:
                            reasons = ["probe_end"]
                        features.append(engine.task._curr_amp_obs_buf[0].cpu().numpy().copy())
                        latents.append(z[0].cpu().numpy().copy())
                        rows.append((values, reasons, q.copy()))
                        if video and step % 2 == 0:
                            label = ("F1 ASE epoch{} | case {}/{} | clip {} {} | {}\n".format(
                                actor.epoch, case+1, len(probes), mid+1,
                                refs.names[mid], mode)
                                + "POLICY output | {} | elapsed={:.2f}s | ref={:.2f}s".format(
                                    "HOLD final z" if step >= len(refs.times[mid]) else "reference-encoder z sequence",
                                    (step+1)*engine.dt, float(refs.times[mid][reference_step])))
                            video.write(engine.render(), label)
                            video_frames += 1
                        if reasons:
                            break
                features = np.asarray(features, dtype=np.float32)
                best, phases, backwards = refs.analyze(features)
                # Exclude first .5s from coverage support: an injected reference
                # state must not count as evidence of policy-maintained behavior.
                warmup = min(25, len(rows))
                if len(rows) > warmup:
                    _, _, backwards = refs.analyze(features[warmup:])
                    for j, d in enumerate(backwards):
                        reverse[mode][j] = np.minimum(reverse[mode][j], d)
                nearest = best.argmin(1)
                target = refs.names[mid] if mid >= 0 else ""
                for step, (values, reasons, q) in enumerate(rows):
                    j = int(nearest[step])
                    writer.writerow([case, mode, target, args.latent_seed+case if mid < 0 else "",
                        step+1, (step+1)*engine.dt, *values, "+".join(reasons), refs.names[j],
                        float(best[step, j]), float(refs.times[j][phases[step, j]]),
                        int(best[step, j] <= refs.thresholds[j]),
                        float(best[step, mid]) if mid >= 0 else "", *q])
                xy = np.array([r[0][:2] for r in rows])
                speeds = np.array([r[0][3:5] for r in rows])
                q_series = np.array([r[2] for r in rows])
                legs = [i for i, n in enumerate(names) if any(s in n for s in ("hip", "knee", "ankle"))]
                valid = best[warmup:]
                nearest_valid = nearest[warmup:]
                hist = {refs.names[j]: int((nearest_valid == j).sum()) for j in range(len(refs.names))}
                summary = dict(case=case, mode=mode, target=target, init=state["init"],
                    latent_seed=args.latent_seed+case if mid < 0 else None,
                    seconds=len(rows)*engine.dt, termination=rows[-1][1],
                    distance_m=float(np.linalg.norm(xy[-1] - state["root"][:2])),
                    mean_vx=float(speeds[:, 0].mean()), mean_speed=float(np.linalg.norm(speeds, axis=1).mean()),
                    leg_peak_to_peak_rad=float(np.ptp(q_series[warmup:] if len(rows)>warmup else q_series, axis=0)[legs].max()),
                    closest_histogram=hist, video_start_s=start_video,
                    video_end_s=video_frames/25 if start_video is not None else None)
                if mid >= 0 and len(valid):
                    tail = speeds[-min(len(speeds), max(1, round(2/engine.dt))):]
                    summary.update(reference_sequence_s=len(refs.times[mid])*engine.dt,
                        tail_2s_mean_speed=float(np.linalg.norm(tail, axis=1).mean()),
                        tail_2s_max_speed=float(np.linalg.norm(tail, axis=1).max()))
                    summary.update(target_distance_median=float(np.median(valid[:, mid])),
                        target_within_threshold_fraction=float((valid[:, mid] <= refs.thresholds[mid]).mean()))
                summaries.append(summary)
                all_features.append(features[1:])
                all_z.append(np.asarray(latents))
                all_ids.append(np.full(len(rows), case, dtype=np.int32))
                f.flush()
                print("COVERAGE_CASE " + json.dumps(summary), flush=True)
        # Compact feature archive permits metric changes without repeating physics.
        np.savez_compressed(args.output / (args.prefix + "coverage_features.npz"), amp=np.concatenate(all_features),
                            z=np.concatenate(all_z), case=np.concatenate(all_ids),
                            reference=np.concatenate([w[:, 0].cpu().numpy() for w in refs.windows]),
                            reference_pair=np.concatenate([p.cpu().numpy() for p in refs.pairs]),
                            reference_counts=np.array([len(t) for t in refs.times]),
                            reference_times=np.concatenate([t.cpu().numpy() for t in refs.times]))
        coverage = []
        for j, name in enumerate(refs.names):
            row = dict(clip=name, threshold=refs.thresholds[j], reference_pairs=len(refs.times[j]))
            for mode, values in reverse.items():
                row[mode+"_reference_fraction"] = float((values[j] <= refs.thresholds[j]).mean())
                row[mode+"_distance_median"] = float(np.median(values[j]))
            coverage.append(row)
        result = dict(checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                      epoch=actor.epoch, seed=args.seed, latent_seed_start=args.latent_seed,
                      random_cases=args.random_cases, deterministic=True, dt=engine.dt,
                      selected_clips=[i+1 for i in selected], requested_duration_s=args.duration,
                      reference_end_behavior="hold final encoded z",
                      random_num_envs=max(1, args.random_cases), encoded_num_envs=1,
                      metric="equal-group standardized 89D AMP consecutive-state RMS",
                      threshold="per-clip self NN p95 excluding +/-0.1s, clipped to [.25,.75]",
                      warmup_excluded_s=.5, control_config=actor.control_config,
                      caveat="heuristic support only; nearest label is not proof of learning; encoder probes are auxiliary",
                      references=refs.lib.get_manifest(), coverage=coverage, cases=summaries,
                      video_seconds=video_frames/25)
        (args.output / (args.prefix + "coverage_summary.json")).write_text(json.dumps(result, indent=2), encoding="utf-8")
        with (args.output / (args.prefix + "coverage_summary.csv")).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(
                key for row in coverage for key in row)))
            writer.writeheader()
            writer.writerows(coverage)
        print("COVERAGE_DONE " + json.dumps(dict(cases=len(summaries), video_seconds=video_frames/25)), flush=True)
        analyze_saved(args.output, prefix=args.prefix)
    finally:
        try:
            if video:
                video.close()
        finally:
            if engine is not None:
                engine.close()


if __name__ == "__main__":
    main()
