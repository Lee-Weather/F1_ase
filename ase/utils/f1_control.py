"""Shared F1 action mapping for training and both replay simulators."""
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET


LEGACY_CONTROL = {"actionMapping": "default_offset", "actionScale": 0.5}
CONTROL_KEYS = ("actionMapping", "actionScale", "actionLimitMargin", "pdTargetLimits", "pdTargetClampJoints", "defaultJointPos",
                "pdGains", "armature", "controlFrequencyInv", "controlNeutralJointPos",
                "legTargetFilterAlpha", "lastActionSource")
CONTROL_DEFAULTS = {"pdTargetLimits": {}, "pdTargetClampJoints": [], "controlNeutralJointPos": {},
                    "legTargetFilterAlpha": 1.0, "lastActionSource": "raw"}


def pd_action_parameters(env, names, lower, upper):
    """Return neutral targets and negative/positive scales in simulator order.

    joint_limits uses the explicit control neutral, or the default posture. Its two
    slopes reach the joint limits plus a fraction of joint travel as PD margin.
    The legacy affine mapping remains available for existing checkpoints.
    """
    overrides = env.get("controlNeutralJointPos", {})
    neutral = np.array([overrides.get(n, env["defaultJointPos"][n]) for n in names], dtype=np.float32)
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
    """Inverse mapping for NumPy arrays or torch tensors."""
    delta = target - neutral
    return delta / (positive * (delta >= 0) + negative * (delta < 0))


def filter_targets(request, previous, alpha):
    """Pure math; only the physics-step caller commits the returned state."""
    return previous + alpha * (request - previous)


def foot_sole_points(urdf):
    """Four actual bottom-mesh vertices per foot in its link frame (metres).

    F1 collision STL files are binary. Apply the URDF mesh scale and origin;
    using real bottom vertices avoids assuming the ankle link is the sole.
    """
    urdf = Path(urdf)
    root, feet = ET.parse(urdf).getroot(), []
    for side in ("left", "right"):
        collision = root.find("link[@name='" + side + "_ankle_roll_link']/collision")
        mesh, origin = collision.find("geometry/mesh"), collision.find("origin")
        path = urdf.parent / mesh.get("filename")
        with path.open("rb") as stream:
            stream.read(80)
            count = int.from_bytes(stream.read(4), "little")
            dtype = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
            points = np.frombuffer(stream.read(), dtype=dtype, count=count)["vertices"].reshape(-1, 3).astype(np.float64)
        points *= np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
        r, p, y = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
        cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
        rotation = np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                             [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])
        points = points @ rotation.T + np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        bottom = points[points[:, 2] <= points[:, 2].min() + .001]
        low, high = bottom[:, :2].min(0), bottom[:, :2].max(0)
        corners = [[x, yy] for x in (low[0], high[0]) for yy in (low[1], high[1])]
        feet.append(np.array([bottom[np.square(bottom[:, :2] - corner).sum(-1).argmin()] for corner in corners]))
    return np.asarray(feet, dtype=np.float32)


def checkpoint_control(checkpoint):
    # Pre-mapping-fix models were all trained with default_offset and scale 0.5.
    control = checkpoint.get("control_config", LEGACY_CONTROL).copy()
    # A missing window means the original full-travel mapping, even when the
    # replay YAML now defines a narrower window for newly trained policies.
    for key, value in CONTROL_DEFAULTS.items():
        control.setdefault(key, value)
    return control
