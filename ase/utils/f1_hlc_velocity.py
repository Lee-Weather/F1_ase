"""Three-axis F1 commands in the horizontal frame of the robot's heading."""


COMMAND_GROUPS = ("stop", "forward", "backward", "left", "right", "turn_left", "turn_right", "mixed")


def sample_velocity_commands(stage, count, device):
    import torch

    groups = torch.multinomial(torch.tensor(stage["commandWeights"], device=device, dtype=torch.float32),
                              count, replacement=True)
    ranges = torch.tensor(stage["movingRanges"], device=device, dtype=torch.float32)
    magnitudes = ranges[:, 0] + torch.rand((count, 4), device=device) * (ranges[:, 1] - ranges[:, 0])
    anchors = stage.get("lowSpeedAnchors", [])
    if anchors:
        anchors = torch.tensor(anchors, device=device)
        picks = anchors[torch.randint(len(anchors), (count, 3), device=device)]
        eligible = (picks >= ranges[:3, 0]) & (picks <= ranges[:3, 1])
        selected = eligible & (torch.rand_like(picks) < stage["anchorProbability"])
        magnitudes[:, :3] = torch.where(selected, picks, magnitudes[:, :3])

    forward, backward, lateral, yaw = magnitudes.unbind(dim=-1)
    if stage.get("yawAnchorProbability", 0.):
        yaw = torch.where(torch.rand_like(yaw) < stage["yawAnchorProbability"], ranges[3, 1], yaw)
    vx = torch.where(groups == 1, forward, torch.where(groups == 2, -backward, 0.))
    vy = torch.where(groups == 3, lateral, torch.where(groups == 4, -lateral, 0.))
    wz = torch.where(groups == 5, yaw, torch.where(groups == 6, -yaw, 0.))

    axes = torch.tensor(stage.get("mixedAxes", [[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]), device=device)
    axes = axes[torch.randint(len(axes), (count,), device=device)]
    signs = torch.rand((count, 3), device=device)
    mixed = torch.stack((torch.where(signs[:, 0] < 1 - stage.get("mixedBackwardProbability", .5), forward, -backward),
                         torch.where(signs[:, 1] < .5, lateral, -lateral),
                         torch.where(signs[:, 2] < .5, yaw, -yaw)), dim=-1) * axes
    command = torch.where((groups == 7)[:, None], mixed, torch.stack((vx, vy, wz), dim=-1))
    command[:, :2] *= stage["maxPlanarSpeed"] / command[:, :2].norm(dim=-1, keepdim=True).clamp_min(stage["maxPlanarSpeed"])
    return command, groups


def heading_velocity(fx, fy, vx, vy, wz):
    """Rotate world XY by inverse heading. Yaw rate is along world vertical."""
    length = (fx * fx + fy * fy).clip(1e-12, None) ** .5
    return ((fx * vx + fy * vy) / length, (fx * vy - fy * vx) / length, wz)


def velocity_task_reward(actual, command, foot_speed_sq, leg_velocity_sq, config):
    """Works with Torch batches and NumPy replay arrays using identical math."""
    vx, vy, yaw = actual[..., 0], actual[..., 1], actual[..., 2]
    tx, ty, tz = command[..., 0], command[..., 1], command[..., 2]
    planar = tx * tx + ty * ty
    translating = planar > 1e-12
    turning = (tz * tz) > 1e-12
    stop = ~translating & ~turning
    scales = config["velocityTrackingSigma"]
    standing = config["standingSpeedSigma"]
    sx = scales[0] + (standing - scales[0]) * stop
    sy = scales[1] + (standing - scales[1]) * stop
    sw = scales[2] + (config["standingYawSigma"] - scales[2]) * stop
    cost = ((vx - tx) / sx) ** 2 + ((vy - ty) / sy) ** 2 + ((yaw - tz) / sw) ** 2
    cost += stop * (config["standFootSpeedScale"] * foot_speed_sq
                    + config["standLegVelocityScale"] * leg_velocity_sq)

    # Reward progress in the requested direction, including pure yaw commands.
    planar_progress = ((vx * tx + vy * ty) / planar.clip(1e-12, None)).clip(0., 1.)
    yaw_progress = (yaw * tz / (tz * tz).clip(1e-12, None)).clip(0., 1.)
    axes = translating * 1. + turning * 1.
    progress = (planar_progress + yaw_progress + stop) / axes.clip(1., None)
    return (.2 + .8 * progress) / (1. + cost)


class VelocityMetrics:
    def __init__(self, device):
        import torch

        self.totals = torch.zeros((len(COMMAND_GROUPS), 10), device=device)

    def step(self, groups, actual, command, valid, quiet, failed):
        import torch

        direction = (((command == 0) | (actual * command > 0)).all(-1)
                     & (command != 0).any(-1))
        samples = torch.cat((valid[:, None], actual, (actual - command).abs(),
                             quiet[:, None], failed[:, None], direction[:, None]), dim=-1)
        self.totals.index_add_(0, groups, samples * valid[:, None])

    def drain(self):
        result = dict(zip(COMMAND_GROUPS, self.totals.clone().unbind()))
        self.totals.zero_()
        return result

    @staticmethod
    def summarize(totals):
        result = {}
        all_samples = sum(float(values[0]) for values in totals.values())
        for group, values in totals.items():
            count, vx, vy, wz, ex, ey, ez, quiet, falls, direction = values.tolist()
            if not count:
                continue
            prefix = "HLC_Velocity/" + group + "/"
            result.update({prefix + name: value / count for name, value in (
                ("vx_m_s", vx), ("vy_m_s", vy), ("wz_rad_s", wz),
                ("vx_mae_m_s", ex), ("vy_mae_m_s", ey), ("wz_mae_rad_s", ez))})
            result[prefix + "sample_fraction"] = count / all_samples
            result[prefix + "falls"] = int(falls)
            if group == "stop":
                result[prefix + "quiet_fraction"] = quiet / count
            else:
                result[prefix + "direction_fraction"] = direction / count
        return result
