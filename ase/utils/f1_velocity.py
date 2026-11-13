"""Shared F1 velocity command rules, task configuration and reward math.

Command helpers work with NumPy or torch tensors. Importing this module does
not require Isaac Gym or torch, so replay inputs can be checked on any host.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml


def load_env_config(path):
    """Load an environment YAML, optionally extending a sibling base YAML."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = cfg.pop("base", None)
    if base is None:
        return cfg
    result = load_env_config(path.parent / base)

    def merge(destination, source):
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(destination.get(key), dict):
                merge(destination[key], value)
            else:
                destination[key] = value
    merge(result, cfg)
    return result


def velocity_settings(env):
    settings = deepcopy(env["velocityTask"])
    amp = env.get("policyAlgorithm", "ASE") == "AMP"
    if amp:
        settings = {key: value for key, value in settings.items() if not key.startswith("encoder")}
    for key in ("maxSpeed", "acceleration", "deceleration", "trackingSigma",
                "lateralSigma", "yawSigma", "tiltSigma", "quietJointSigma", "quietFootSigma"):
        if settings[key] <= 0:
            raise ValueError("velocityTask." + key + " must be positive")
    if not 0 <= settings["deadzone"] < settings["minMovingSpeed"] <= settings["maxSpeed"]:
        raise ValueError("Velocity deadzone/minimum/maximum are inconsistent")
    if not amp and not 0 <= settings["encoderZeroSpeed"] < settings["encoderFullSpeed"]:
        raise ValueError("Encoder speed gate thresholds are inconsistent")
    if amp and not 0 <= settings["standCommandZeroSpeed"] < settings["standCommandFullSpeed"]:
        raise ValueError("Standing command thresholds are inconsistent")
    for key in ("standDuration", "walkDuration"):
        low, high = settings[key]
        if not 0 < low <= high:
            raise ValueError("Invalid " + key)
    if settings.get("version", 1) >= 2:
        if not amp and not 0 <= settings["encoderMotionZeroSpeed"] < settings["encoderMotionFullSpeed"]:
            raise ValueError("Encoder motion thresholds are inconsistent")
        if settings["progressWindowSeconds"] <= 0:
            raise ValueError("Progress window must be positive")
        if not 0 < settings["standPose"]["centralFraction"] <= 1:
            raise ValueError("Standing reference centralFraction must be in (0, 1]")
        if not np.isclose(np.linalg.norm(settings["footForwardAxis"]), 1):
            raise ValueError("Foot forward axis must be a unit vector")
        for stage in settings["curriculum"].values():
            low, high = stage["movingSpeed"]
            if not settings["minMovingSpeed"] <= low <= high <= settings["maxSpeed"]:
                raise ValueError("Curriculum speed outside command range")
            if not 0 <= stage["referenceProbability"] <= 1:
                raise ValueError("Invalid reference initialization probability")
    if settings.get("version", 1) >= 4:
        if not amp or env.get("lastActionSource") != "executed" or not 0 < env.get("legTargetFilterAlpha", 0) <= 1:
            raise ValueError("Velocity v4 requires AMP and observable filtered targets")
        if not 0 <= settings["contactExitForce"] < settings["contactEnterForce"]:
            raise ValueError("Contact force hysteresis is inconsistent")
        for key in ("contactConfirmSeconds", "soleMaxHeight", "solePointBand", "slipSpeedScale",
                    "targetRateFloor", "swingMinSeconds", "swingMinClearance"):
            if settings[key] <= 0:
                raise ValueError("velocityTask." + key + " must be positive")
    return settings


def reference_target_scales(motion_file, names, alpha, dt, floor, diagnostics=None):
    """P99 request-to-previous-execution rates, using genuine reference timing.

    Each clip starts at its first pose; no artificial jump across clip boundaries.
    The returned named scales are saved in the policy, so replay needs no re-fit.
    """
    path = Path(motion_file)
    entries = yaml.safe_load(path.read_text())["motions"]
    rates = []
    for entry in entries:
        with np.load(path.parent / entry["file"], allow_pickle=False) as raw:
            order = [raw["joint_names"].tolist().index(name) for name in names]
            q = raw["joint_pos"][:, order].astype(np.float64)
            source_t = np.arange(len(q)) / float(raw["fps"].reshape(-1)[0])
            times = np.arange(0, source_t[-1] + 1e-8, dt)
            q = np.stack([np.interp(times, source_t, col) for col in q.T], axis=-1)
        previous = q[0].copy()
        executed = [previous.copy()] if diagnostics is not None else None
        for request in q[1:]:
            rates.append(np.abs(request - previous) / dt)
            previous += alpha * (request - previous)
            if executed is not None:
                executed.append(previous.copy())
        if diagnostics is not None:
            actual = np.asarray(executed)
            legs = [i for i, name in enumerate(names) if any(p in name for p in ("hip", "knee", "ankle"))]
            diagnostics[entry["file"]] = {
                name: {"tracking_rms_deg": round(float(np.rad2deg(np.sqrt(np.mean((actual[:, i] - q[:, i])**2)))), 4),
                       "range_retained": round(float(np.ptp(actual[:, i]) / max(np.ptp(q[:, i]), 1e-8)), 4)}
                for i, name in enumerate(names) if i in legs}
    scales = np.maximum(np.quantile(rates, .99, axis=0), floor)
    return {name: round(float(value), 6) for name, value in zip(names, scales)}


def stand_command_threshold(settings):
    return settings.get("standCommandZeroSpeed", settings.get("encoderZeroSpeed", .05))


def stop_quiet(command, speed, foot_speed, leg_speed, settings):
    """Identical instantaneous stop criterion for Gym and offline replay."""
    return ((command <= stand_command_threshold(settings)) & (speed < .05)
            & (foot_speed < .1) & (leg_speed < .3))


def _clip(value, low, high):
    return value.clamp(low, high) if hasattr(value, "clamp") else np.clip(value, low, high)


def ramp_commands(current, target, dt, settings):
    """One 50 Hz command update; only forward motion is enabled in this task."""
    result = current.copy() if isinstance(current, np.ndarray) else current.clone()
    forward = _clip(target[..., 0], 0.0, settings["maxSpeed"])
    forward = forward * (forward >= settings["deadzone"])
    delta = _clip(forward - current[..., 0], -settings["deceleration"] * dt,
                  settings["acceleration"] * dt)
    result[..., 0] = current[..., 0] + delta
    result[..., 1:] = 0
    return result


def encoder_command_mask(commands, settings):
    low, high = settings["encoderZeroSpeed"], settings["encoderFullSpeed"]
    return _clip((abs(commands[..., 0]) - low) / (high - low), 0.0, 1.0)


def encoder_motion_mask(commands, forward_speed, history_valid, settings):
    """A transition gate shared by encoder reward and encoder minibatch loss."""
    low, high = settings["encoderMotionZeroSpeed"], settings["encoderMotionFullSpeed"]
    return (encoder_command_mask(commands, settings) * history_valid
            * _clip((forward_speed - low) / (high - low), 0.0, 1.0))


def foot_geometry(root_rotation, foot_rotation, forward_axis):
    """Signed foot heading in the root's horizontal frame; sole tilt to world Z."""
    import torch
    from isaacgym.torch_utils import quat_rotate
    from utils import torch_utils
    forward = torch.zeros_like(foot_rotation[..., :3])
    forward[:] = torch.as_tensor(forward_axis, device=forward.device)
    direction = quat_rotate(foot_rotation.reshape(-1, 4), forward.reshape(-1, 3))
    heading = torch_utils.calc_heading_quat_inv(root_rotation)
    heading = heading[:, None, :].expand_as(foot_rotation).reshape(-1, 4)
    direction = quat_rotate(heading, direction).reshape_as(forward)
    yaw = torch.atan2(direction[..., 1], direction[..., 0])
    up = torch.zeros_like(forward)
    up[..., 2] = 1
    up = quat_rotate(foot_rotation.reshape(-1, 4), up.reshape(-1, 3)).reshape_as(up)
    return yaw, torch.acos(up[..., 2].clamp(-1, 1))


def masked_mean(values, mask):
    """Differentiable weighted mean, including an all-standing minibatch."""
    mask = mask.reshape(values.shape).to(dtype=values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1e-6)


def task_rewards(velocity, yaw_rate, tilt, leg_velocity, foot_speed, action,
                 previous_action, torque, effort, joint_pos, lower, upper, commands, settings,
                 posture=None):
    """Privileged quantities are used for rewards only, never actor inputs."""
    import torch
    weights = settings["rewardWeights"]
    full_speed = settings.get("standCommandFullSpeed", settings.get("encoderFullSpeed", .15))
    stop_gate = (1 - commands[:, 0].abs() / full_speed).clamp(0, 1)
    stop_gate *= (1 - velocity[:, :2].norm(dim=-1) / settings["stopSpeedGate"]).clamp(0, 1)
    quiet = torch.exp(-(leg_velocity / settings["quietJointSigma"]).square().mean(-1)
                      -(foot_speed / settings["quietFootSigma"]).square().mean(-1))
    outside = (joint_pos - torch.max(lower, torch.min(joint_pos, upper))) / (upper - lower).clamp_min(1e-3)
    components = {
        "lin_vel_tracking": torch.exp(-((velocity[:, 0] - commands[:, 0]) / settings["trackingSigma"]).square()),
        "lateral_tracking": torch.exp(-((velocity[:, 1] - commands[:, 1]) / settings["lateralSigma"]).square()),
        "yaw_tracking": torch.exp(-((yaw_rate - commands[:, 2]) / settings["yawSigma"]).square()),
        "upright": torch.exp(-(tilt / settings["tiltSigma"]).square()),
        "stand_quiet": stop_gate * quiet,
        "action_rate": -(action - previous_action).square().mean(-1),
        "torque": -(torque / effort.clamp_min(1)).square().mean(-1),
        "joint_limits": -outside.square().mean(-1),
    }
    if settings.get("version", 1) >= 2:
        if posture is None:
            raise ValueError("Velocity v2 requires foot geometry and PD targets")
        radians = np.pi / 180
        components.update({
            "lin_vel_tracking": 1 / (1 + ((velocity[:, 0] - commands[:, 0]) / settings["trackingSigma"]).square()),
            "lateral_tracking": -(velocity[:, 1] / settings["lateralSigma"]).square(),
            "yaw_tracking": -(yaw_rate / settings["yawSigma"]).square(),
            "upright": -(tilt / settings["tiltSigma"]).square(),
            "stand_quiet": stop_gate * quiet * torch.exp(-posture["stand_pose_error"]),
            "foot_heading": -((posture["foot_yaw"].abs() - settings["footHeadingToleranceDeg"] * radians).clamp_min(0)
                               / (settings["footHeadingScaleDeg"] * radians)).square().mean(-1),
            "hip_yaw": -((posture["hip_yaw"].abs() - settings["hipYawToleranceDeg"] * radians).clamp_min(0)
                          / (settings["hipYawScaleDeg"] * radians)).square().mean(-1),
            "stance_foot_tilt": -(((posture["foot_tilt"] - settings["stanceTiltToleranceDeg"] * radians).clamp_min(0)
                                    / (settings["stanceTiltScaleDeg"] * radians)).square()
                                  * posture["stance"]).mean(-1),
        })
        target = posture["pd_targets"]
        target_outside = (target - torch.max(lower, torch.min(target, upper))) / (upper - lower).clamp_min(1e-3)
        if settings.get("version", 1) >= 3:
            # Raw targets retain a learning signal even when execution clamps
            # the knees. Average over the knees, not all 29 joints.
            knees = posture["knee_ids"]
            others = [i for i in range(target.shape[-1]) if i not in knees]
            components["joint_limits"] -= target_outside[:, others].square().mean(-1)
            components["joint_limits"] -= target_outside[:, knees].square().mean(-1)
        else:
            components["joint_limits"] -= target_outside.square().mean(-1)
    if settings.get("version", 1) >= 4:
        components["action_rate"] = -(action[:, posture["upper_ids"]]
                                        - previous_action[:, posture["upper_ids"]]).square().mean(-1)
        components["leg_target_rate"] = -posture["normalized_target_jump"].square().mean(-1)
        components["feet_slip"] = -((posture["sole_speed"] / settings["slipSpeedScale"]).square()
                                      * posture["stance"]).mean(-1)
    weighted = {key: weights[key] * value for key, value in components.items()}
    return sum(weighted.values()), weighted


def scripted_speed(seconds, profile="walk-stop"):
    if profile == "walk":
        return 0.0 if seconds < .5 else .5
    schedule = ([(5, 0), (13, .3), (21, .6), (29, .8), (35, .3), (45, 0)]
                if profile == "walk-stop" else
                [(5, 0), (13, .6), (23, 0), (33, .3), (45, 0)])
    for end, speed in schedule:
        if seconds < end:
            return speed
    return 0.0


def policy_spec(settings, dt, motion_file, algorithm="ASE"):
    """Versioned deployment-relevant schema stored with the trained policy."""
    spec = {
        "version": settings.get("version", 1), "task": "HumanoidF1Velocity", "actor_obs": 96, "critic_obs": 451,
        "amp_obs": 890, "actions": 29, "latent_dim": 32, "latent_hold_episode": True,
        "actor_fields": [["base_ang_vel", 3], ["projected_gravity", 3],
                         ["joint_pos_rel", 29], ["joint_vel", 29], ["last_action", 29],
                         ["velocity_commands", 3]],
        "command_frame": "horizontal_heading", "command_units": ["m/s", "m/s", "rad/s"],
        "command_scale": [1.0, 1.0, 1.0], "control_dt": float(dt),
        "velocity_settings": deepcopy(settings), "motion_file": str(motion_file),
    }
    if algorithm == "AMP":
        spec.update(version=max(3, settings["version"]), algorithm="AMP", task_reward_version=settings["version"], latent_dim=0)
        spec.pop("latent_hold_episode")
        if settings["version"] >= 4:
            spec["critic_obs"] = 480
            spec["actor_fields"][4][0] = "last_executed_action"
            spec["critic_extra_fields"] = [["last_executed_action", 29]]
    return spec
