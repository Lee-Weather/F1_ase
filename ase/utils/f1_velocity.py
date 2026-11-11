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
    for key in ("maxSpeed", "acceleration", "deceleration", "trackingSigma",
                "lateralSigma", "yawSigma", "tiltSigma", "quietJointSigma", "quietFootSigma"):
        if settings[key] <= 0:
            raise ValueError("velocityTask." + key + " must be positive")
    if not 0 <= settings["deadzone"] < settings["minMovingSpeed"] <= settings["maxSpeed"]:
        raise ValueError("Velocity deadzone/minimum/maximum are inconsistent")
    if not 0 <= settings["encoderZeroSpeed"] < settings["encoderFullSpeed"]:
        raise ValueError("Encoder speed gate thresholds are inconsistent")
    for key in ("standDuration", "walkDuration"):
        low, high = settings[key]
        if not 0 < low <= high:
            raise ValueError("Invalid " + key)
    return settings


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


def masked_mean(values, mask):
    """Differentiable weighted mean, including an all-standing minibatch."""
    mask = mask.reshape(values.shape).to(dtype=values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1e-6)


def task_rewards(velocity, yaw_rate, tilt, leg_velocity, foot_speed, action,
                 previous_action, torque, effort, joint_pos, lower, upper, commands, settings):
    """Privileged quantities are used for rewards only, never actor inputs."""
    import torch
    weights = settings["rewardWeights"]
    stop_gate = (1 - commands[:, 0].abs() / settings["encoderFullSpeed"]).clamp(0, 1)
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
    weighted = {key: weights[key] * value for key, value in components.items()}
    return sum(weighted.values()), weighted


def scripted_speed(seconds, profile="walk-stop"):
    schedule = ([(5, 0), (13, .3), (21, .6), (29, .8), (35, .3), (45, 0)]
                if profile == "walk-stop" else
                [(5, 0), (13, .6), (23, 0), (33, .3), (45, 0)])
    for end, speed in schedule:
        if seconds < end:
            return speed
    return 0.0


def policy_spec(settings, dt, motion_file):
    """Versioned deployment-relevant schema stored with the trained policy."""
    return {
        "version": 1, "task": "HumanoidF1Velocity", "actor_obs": 96, "critic_obs": 451,
        "amp_obs": 890, "actions": 29, "latent_dim": 32, "latent_hold_episode": True,
        "actor_fields": [["base_ang_vel", 3], ["projected_gravity", 3],
                         ["joint_pos_rel", 29], ["joint_vel", 29], ["last_action", 29],
                         ["velocity_commands", 3]],
        "command_frame": "horizontal_heading", "command_units": ["m/s", "m/s", "rad/s"],
        "command_scale": [1.0, 1.0, 1.0], "control_dt": float(dt),
        "velocity_settings": deepcopy(settings), "motion_file": str(motion_file),
    }
