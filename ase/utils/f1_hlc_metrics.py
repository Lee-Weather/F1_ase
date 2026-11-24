"""Observation-only, physical-step HLC telemetry; no reward or policy changes."""
import math

import torch


class HLCMetrics:
    SWING_PROGRESS = tuple(f"{side}_{phase}_progress" for side in ("left", "right")
                           for phase in ("unload", "lift", "advance"))
    SPEED_BANDS = (("zero", 0., 0.), ("0_0p5", 0., .5),
                   ("0p5_1", .5, 1.), ("1_1p5", 1., 1.5), ("1p5_up", 1.5, float("inf")),
                   ("near_0p1", 1e-6, .15), ("near_0p2", .15, .25), ("near_0p3", .25, .35))

    def __init__(self, count, device, dt, config):
        self.count, self.device, self.dt = count, device, dt
        self.config = config
        # Isaac Gym exposes a float32 dt (e.g. 0.01999999955); do not turn
        # an exact 25/250-step window into 26/251 steps through ceil noise.
        steps = lambda seconds: math.ceil(seconds / dt - 1e-5)
        self.tail_steps = steps(5.)
        self.quiet_steps = steps(.5)
        self.restart_steps = steps(2.)
        self.deadline_steps = steps(3.)
        self.initialized = torch.zeros(count, dtype=torch.bool, device=device)
        self.mode = torch.zeros(count, dtype=torch.long, device=device)  # 1 stop, 2 restart
        self.age = torch.zeros_like(self.mode)
        self.run = torch.zeros_like(self.mode)
        self.reached = torch.full_like(self.mode, -1)
        self.last_target = torch.zeros(count, device=device)
        self.last_planar = torch.zeros_like(self.last_target)
        self.last_quiet = torch.zeros_like(self.initialized)
        self.path = torch.zeros((self.tail_steps, count), device=device)
        self.tail_path = torch.zeros_like(self.last_target)
        self.cursor = 0
        keys = ["{}/{}".format(band, name) for band, _, _ in self.SPEED_BANDS
                for name in ("samples", "speed_error_sum", "lateral_speed_sum")]
        keys += ["zero/quiet_samples"]
        keys += ["{}/{}".format(kind, name) for kind in ("stop", "restart")
                 for name in ("requested", "unqualified", "completed", "successes", "failures",
                              "truncated", "success_delay_sum_s")]
        if config.get("gaitReward") is not None:
            keys += ["gait/" + name for name in ("moving_samples", "left_footfalls", "right_footfalls",
                                                 "alternating_footfalls", "recent_alternation_samples",
                                                 "stance_samples", "stance_slip_sum_m_s")]
            if config["gaitReward"].get("swingProgressWeight", 0.) > 0:
                keys += ["gait/" + name for name in self.SWING_PROGRESS]
        self.totals = {key: torch.zeros((), device=device) for key in keys}

    def _add(self, key, values):
        self.totals[key] += values.float().sum()

    def record_gait(self, live, recent, alternating, complete, slip, stance, swing_increment=None):
        self._add("gait/moving_samples", live)
        self._add("gait/left_footfalls", complete[:, 0])
        self._add("gait/right_footfalls", complete[:, 1])
        self._add("gait/alternating_footfalls", alternating)
        self._add("gait/recent_alternation_samples", recent > 0)
        self._add("gait/stance_samples", stance)
        self._add("gait/stance_slip_sum_m_s", slip)
        if swing_increment is not None:
            sums = swing_increment.sum(dim=0).flatten()
            for name, value in zip(self.SWING_PROGRESS, sums):
                self.totals["gait/" + name] += value

    def _clear(self, ids):
        self.mode[ids] = 0
        self.age[ids] = 0
        self.run[ids] = 0
        self.reached[ids] = -1
        self.tail_path[ids] = 0
        self.path[:, ids] = 0

    def _finish(self, ids, failed=None, interrupted=False):
        """Command changes complete events; timeouts/resets censor them, falls fail."""
        if failed is None:
            failed = torch.zeros_like(ids)
        good_stop = (ids & (self.mode == 1) & (self.age >= self.tail_steps)
                     & (self.reached >= 0) & (self.reached <= self.deadline_steps)
                     & (self.tail_path <= .1) & ~failed)
        restart_ready = good_stop & self.last_quiet
        for kind, mode in (("stop", 1), ("restart", 2)):
            selected = ids & (self.mode == mode)
            success = good_stop if mode == 1 else selected & (self.reached >= 0) & ~failed
            enough = self.age >= (self.tail_steps if mode == 1 else
                                  self.deadline_steps + self.restart_steps)
            truncated = selected & ~failed & (torch.ones_like(ids) if interrupted else ~enough & ~success)
            completed = selected & ~truncated
            success = success & completed
            self._add(kind + "/completed", completed)
            self._add(kind + "/successes", success)
            self._add(kind + "/failures", completed & ~success)
            self._add(kind + "/truncated", truncated)
            self._add(kind + "/success_delay_sum_s", self.reached.clamp(min=0) * self.dt * success)
        self._clear(ids)
        return restart_ready

    def reset(self, env_ids):
        ids = torch.zeros_like(self.initialized)
        ids[env_ids] = True
        self._finish(ids, interrupted=True)
        self.initialized[ids] = False
        self.last_quiet[ids] = False

    @torch.no_grad()
    def step(self, target, vx, vy, foot_rms, leg_rms, valid, failed, done):
        planar = torch.sqrt(vx.square() + vy.square())
        quiet = ((planar < self.config.get("quietPlanarSpeed", .05))
                 & (foot_rms < self.config.get("quietFootSpeed", .1))
                 & (leg_rms < self.config.get("quietLegVelocity", .3)) & ~failed)
        zero = target <= 1e-6
        for band, low, high in self.SPEED_BANDS:
            selected = valid & (zero if band == "zero" else ~zero & (target >= low) & (target < high))
            self._add(band + "/samples", selected)
            self._add(band + "/speed_error_sum", (vx - target).abs() * selected)
            self._add(band + "/lateral_speed_sum", vy.abs() * selected)
        self._add("zero/quiet_samples", valid & zero & quiet)

        changed = valid & self.initialized & ((target - self.last_target).abs() > 1e-6)
        restart_ready = self._finish(changed)
        stop = changed & zero & (self.last_target > 1e-6)
        restart = changed & ~zero & (self.last_target <= 1e-6)
        moving_stop = stop & (self.last_planar >= .1)
        qualified_restart = restart & restart_ready
        self._add("stop/requested", stop)
        self._add("stop/unqualified", stop & ~moving_stop)
        self._add("restart/requested", restart)
        self._add("restart/unqualified", restart & ~qualified_restart)
        self.mode[moving_stop] = 1
        self.mode[qualified_restart] = 2
        active = valid & (self.mode > 0)
        self.age += active.long()
        stopping = active & (self.mode == 1)
        # At targets below 0.1m/s, a stationary robot is inside +/-0.1.
        # Require actual forward progress before counting a restart window.
        in_band = ((vx - target).abs() <= .1) & (vx >= .5 * target) & ~failed
        sustained = (stopping & quiet) | (active & (self.mode == 2) & in_band)
        self.run = torch.where(active, torch.where(sustained, self.run + 1, torch.zeros_like(self.run)), self.run)
        first_stop = stopping & (self.reached < 0) & (self.run >= self.quiet_steps)
        self.reached[first_stop] = self.age[first_stop]
        entry = self.age - self.run + 1
        first_restart = (active & (self.mode == 2) & (self.reached < 0)
                         & (self.run >= self.restart_steps) & (entry <= self.deadline_steps))
        self.reached[first_restart] = entry[first_restart]
        increment = planar * self.dt * stopping
        self.tail_path += increment - self.path[self.cursor]
        self.tail_path[~stopping] = 0
        self.path[self.cursor].copy_(increment)
        self.cursor = (self.cursor + 1) % self.tail_steps
        self.last_quiet[valid] = quiet[valid]
        self.last_planar[valid] = planar[valid]
        self.last_target[valid] = target[valid]
        self.initialized |= valid
        self._finish(valid & done, failed=valid & failed, interrupted=True)

    def drain(self):
        result = {key: value.clone() for key, value in self.totals.items()}
        for value in self.totals.values():
            value.zero_()
        return result

    def pending(self):
        return {"stop": int((self.mode == 1).sum()), "restart": int((self.mode == 2).sum())}


def summarize_hlc_metrics(totals):
    """Convert interval sums into weighted rates, never averaging absent bands."""
    result = {}
    for band, _, _ in HLCMetrics.SPEED_BANDS:
        count = totals.get(band + "/samples", 0.)
        result["HLC_Speed/" + band + "/samples"] = count
        if count:
            result["HLC_Speed/" + band + "/mae_m_s"] = totals[band + "/speed_error_sum"] / count
            result["HLC_Speed/" + band + "/lateral_abs_m_s"] = totals[band + "/lateral_speed_sum"] / count
    if totals.get("zero/samples", 0):
        result["HLC_Events/zero_quiet_fraction"] = totals["zero/quiet_samples"] / totals["zero/samples"]
    for kind in ("stop", "restart"):
        for name in ("requested", "unqualified", "completed", "successes", "failures", "truncated"):
            result["HLC_Events/" + kind + "_" + name] = totals.get(kind + "/" + name, 0.)
        completed = totals.get(kind + "/completed", 0.)
        successes = totals.get(kind + "/successes", 0.)
        if completed:
            result["HLC_Events/" + kind + "_success_rate"] = successes / completed
        if successes:
            result["HLC_Events/" + kind + "_success_delay_mean_s"] = totals[kind + "/success_delay_sum_s"] / successes
    if "gait/moving_samples" in totals:
        for name in ("left_footfalls", "right_footfalls", "alternating_footfalls", "moving_samples", "stance_samples"):
            result["HLC_Gait/" + name] = totals["gait/" + name]
        if totals["gait/moving_samples"]:
            result["HLC_Gait/recent_alternation_fraction"] = (
                totals["gait/recent_alternation_samples"] / totals["gait/moving_samples"])
        if totals["gait/stance_samples"]:
            result["HLC_Gait/stance_slip_mean_m_s"] = totals["gait/stance_slip_sum_m_s"] / totals["gait/stance_samples"]
        for name in HLCMetrics.SWING_PROGRESS:
            if "gait/" + name in totals:
                result["HLC_Gait/" + name] = totals["gait/" + name]
    return result
