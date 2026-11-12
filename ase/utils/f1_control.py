"""Shared F1 action mapping for training and both replay simulators."""
import numpy as np


LEGACY_CONTROL = {"actionMapping": "default_offset", "actionScale": 0.5}
CONTROL_KEYS = ("actionMapping", "actionScale", "actionLimitMargin", "pdTargetLimits", "defaultJointPos",
                "pdGains", "armature", "controlFrequencyInv")


def pd_action_parameters(env, names, lower, upper):
    """Return neutral targets and negative/positive scales in simulator order.

    joint_limits preserves action=0 at the original default posture. Its two
    slopes reach the joint limits plus a fraction of joint travel as PD margin.
    The legacy affine mapping remains available for existing checkpoints.
    """
    neutral = np.array([env["defaultJointPos"][n] for n in names], dtype=np.float32)
    mode = env.get("actionMapping", "default_offset")
    if mode == "default_offset":
        scale = np.full_like(neutral, env["actionScale"])
        return neutral, scale, scale.copy()
    if mode != "joint_limits":
        raise ValueError("Unknown F1 actionMapping: " + str(mode))
    lower, upper = np.array(lower, dtype=np.float32, copy=True), np.array(upper, dtype=np.float32, copy=True)
    margin = float(env.get("actionLimitMargin", 0.1))
    if margin < 0 or (upper <= lower).any() or (neutral < lower - 1e-5).any() or (neutral > upper + 1e-5).any():
        raise ValueError("Invalid joint limits, default posture or actionLimitMargin")
    extra = margin * (upper - lower)
    for name, limits in env.get("pdTargetLimits", {}).items():
        index = names.index(name)
        low, high = map(float, limits)
        if not lower[index] <= low < neutral[index] < high <= upper[index]:
            raise ValueError("Invalid PD target window for " + name)
        lower[index], upper[index], extra[index] = low, high, 0
    negative, positive = neutral - lower + extra, upper - neutral + extra
    if (negative <= 0).any() or (positive <= 0).any():
        raise ValueError("Both action directions require positive scales")
    return neutral, negative, positive


def action_to_targets(action, neutral, negative, positive):
    """Works with NumPy arrays or torch tensors; callers clip actions to [-1, 1]."""
    return neutral + action * (positive * (action >= 0) + negative * (action < 0))


def targets_to_actions(target, neutral, negative, positive):
    """Inverse mapping for NumPy reference data and control checks."""
    delta = target - neutral
    return delta / np.where(delta >= 0, positive, negative)


def checkpoint_control(checkpoint):
    # Pre-mapping-fix models were all trained with default_offset and scale 0.5.
    control = checkpoint.get("control_config", LEGACY_CONTROL).copy()
    # A missing window means the original full-travel mapping, even when the
    # replay YAML now defines a narrower window for newly trained policies.
    control.setdefault("pdTargetLimits", {})
    return control
