"""Velocity-conditioned AMP/ASE with hardware-compatible actor observations."""
from pathlib import Path

import torch
from isaacgym.torch_utils import quat_rotate, quat_rotate_inverse

from env.tasks.humanoid_f1 import HumanoidF1
from utils import torch_utils
from utils.f1_velocity import (encoder_command_mask, encoder_motion_mask, foot_geometry, policy_spec, ramp_commands,
                               task_rewards, velocity_settings, stand_command_threshold, stop_quiet)


class HumanoidF1Velocity(HumanoidF1):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.velocity_settings = velocity_settings(cfg["env"])
        self._algorithm = cfg["env"].get("policyAlgorithm", "ASE")
        self._use_encoder = self._algorithm == "ASE"
        self._velocity_v2 = self.velocity_settings.get("version", 1) >= 2
        self._default_stage = cfg["env"].get("velocityDefaultStage", "A")
        self._curriculum_stage = cfg["env"].get("velocityStageOverride") or self._default_stage
        self._external_velocity_control = False
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        self.commands = torch.zeros((self.num_envs, 3), device=self.device)
        self.command_targets = torch.zeros_like(self.commands)
        self._reward_commands = torch.zeros_like(self.commands)
        self._reward_targets = torch.zeros_like(self.commands)
        self._previous_action = torch.zeros_like(self.actions)
        if self._use_encoder:
            self._encoder_mask = torch.zeros((self.num_envs, 1), device=self.device)
        self._standing_phase = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._command_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._retarget_steps = torch.full_like(self._command_steps, -1)
        self._leg_ids = torch.tensor([i for i, name in enumerate(self._dof_names)
                                     if any(part in name for part in ("hip", "knee", "ankle"))], device=self.device)
        self._feet_ids = self._build_key_body_ids_tensor(["left_ankle_roll_link", "right_ankle_roll_link"])
        self._knee_ids = [i for i, name in enumerate(self._dof_names) if "knee_pitch" in name]
        self._stop_elapsed = torch.full((self.num_envs,), -1.0, device=self.device)
        self._stop_hold = torch.zeros(self.num_envs, device=self.device)
        self._stop_reported = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self._velocity_v2:
            self._hip_yaw_ids = torch.tensor([self._dof_names.index(side + "_hip_yaw_joint")
                                             for side in ("left", "right")], device=self.device)
            self._upper_ids = torch.tensor([i for i in range(self.num_actions) if i not in self._leg_ids.tolist()], device=self.device)
            window = max(1, round(self.velocity_settings["progressWindowSeconds"] / self.dt))
            self._progress_positions = self._humanoid_root_states[:, :3].unsqueeze(0).repeat(window, 1, 1)
            self._progress_cursor = 0
            self._forward_progress_speed = torch.zeros(self.num_envs, device=self.device)
            self._steady_steps = torch.zeros_like(self._command_steps)
            self._retargeted = torch.zeros_like(self._standing_phase)
            pose_cfg = self.velocity_settings["standPose"]
            index = self._motion_index(pose_cfg["motion"])
            start, count = int(self._motion_lib._starts[index]), int(self._motion_lib._motion_num_frames[index])
            trim = int(count * (1 - pose_cfg["centralFraction"]) / 2)
            stable = self._motion_lib._data["joint_pos"][start + trim:start + count - trim]
            self._stand_pose = stable.median(dim=0).values
            self.set_curriculum_stage(self._curriculum_stage)

    def _motion_index(self, name):
        matches = [i for i, file in enumerate(self._motion_lib._motion_files) if Path(file).name == name]
        if len(matches) != 1:
            raise ValueError("Expected one reference motion named " + name)
        return matches[0]

    def set_curriculum_stage(self, stage):
        """Explicit stage selection; no automatic advancement by epoch."""
        if not self._velocity_v2 or stage not in self.velocity_settings["curriculum"]:
            raise ValueError("Unsupported velocity curriculum stage: " + str(stage))
        self._curriculum_stage = stage
        self._hybrid_init_prob = self.velocity_settings["curriculum"][stage]["referenceProbability"]

    def get_training_state(self):
        return {"velocity_stage": self._curriculum_stage} if self._velocity_v2 else {}

    def set_training_state(self, state):
        if self._velocity_v2:
            stage = self.cfg["env"].get("velocityStageOverride") or state.get("velocity_stage", self._default_stage)
            self.set_curriculum_stage(stage)

    def _reset_ref_state_init(self, env_ids):
        stage = self.velocity_settings.get("curriculum", {}).get(self._curriculum_stage, {})
        reference = stage.get("referenceMotion") if self._velocity_v2 else None
        if reference is None or self._external_velocity_control:
            return super()._reset_ref_state_init(env_ids)
        motion_ids = torch.full_like(env_ids, self._motion_index(reference))
        # Keep enough genuine past frames for the AMP history.
        history = (self._num_amp_obs_steps - 1) * self.dt
        lengths = self._motion_lib.get_motion_length(motion_ids)
        times = history + torch.rand(len(env_ids), device=self.device) * (lengths - history).clamp_min(0)
        state = self._motion_lib.get_motion_state(motion_ids, times)
        self._set_env_state(env_ids, *state[:6])
        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids, self._reset_ref_motion_times = motion_ids, times

    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)
        self._num_obs = 96
        self.cfg["env"]["numStates"] = 451

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        self.gym.enable_actor_dof_force_sensors(env_ptr, self.humanoid_handles[-1])

    def _write_commands(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.obs_buf[ids, 93:96] = self.commands[ids]
        self.states_buf[ids, 448:451] = self.commands[ids]

    def _compute_observations(self, env_ids=None):
        super()._compute_observations(env_ids)
        self._write_commands(env_ids)

    def _sample_duration(self, env_ids):
        cfg = self.velocity_settings
        stand, walk = cfg["standDuration"], cfg["walkDuration"]
        low = torch.where(self._standing_phase[env_ids], stand[0], walk[0])
        high = torch.where(self._standing_phase[env_ids], stand[1], walk[1])
        duration = low + torch.rand(len(env_ids), device=self.device) * (high - low)
        self._command_steps[env_ids] = torch.ceil(duration / self.dt).long()
        self._retarget_steps[env_ids] = torch.where(
            self._standing_phase[env_ids], -1, self._command_steps[env_ids] // 2)
        if self._velocity_v2:
            self._steady_steps[env_ids] = 0
            self._retargeted[env_ids] = False
            self._retarget_steps[env_ids] = -1

    def _reset_envs(self, env_ids):
        if len(env_ids) == 0:
            return
        self.commands[env_ids] = 0
        self.command_targets[env_ids] = 0
        self._reward_commands[env_ids] = 0
        self._reward_targets[env_ids] = 0
        self._previous_action[env_ids] = 0
        if self._use_encoder:
            self._encoder_mask[env_ids] = 0
        self._standing_phase[env_ids] = True
        self._stop_elapsed[env_ids] = -1
        self._stop_hold[env_ids] = 0
        self._stop_reported[env_ids] = False
        self._sample_duration(env_ids)
        super()._reset_envs(env_ids)
        if self._velocity_v2:
            self._progress_positions[:, env_ids] = self._humanoid_root_states[env_ids, :3]
            self._forward_progress_speed[env_ids] = 0
            if not self._external_velocity_control:
                stage = self.velocity_settings["curriculum"][self._curriculum_stage]
                low, high = stage["initialStandDuration"]
                self._command_steps[env_ids] = torch.ceil((low + torch.rand(len(env_ids), device=self.device) * (high - low)) / self.dt).long()
                refs = self._reset_ref_env_ids
                if len(refs):
                    heading = torch_utils.calc_heading_quat_inv(self._humanoid_root_states[refs, 3:7])
                    speed = quat_rotate(heading, self._humanoid_root_states[refs, 7:10])[:, 0].clamp(0, self.velocity_settings["maxSpeed"])
                    speed *= speed >= self.velocity_settings["deadzone"]
                    self.commands[refs, 0] = speed
                    moving = refs[speed >= self.velocity_settings["minMovingSpeed"]]
                    self._standing_phase[moving] = False
                    self.command_targets[moving, 0] = self._sample_moving_speed(len(moving))
                    self._sample_duration(moving)
                self._write_commands(env_ids)

    def _sample_moving_speed(self, count):
        low, high = self.velocity_settings["curriculum"][self._curriculum_stage]["movingSpeed"]
        return low + torch.rand(count, device=self.device) * (high - low)

    def _start_stop_timer(self, stopping):
        self._stop_elapsed[stopping] = 0
        self._stop_hold[stopping] = 0
        self._stop_reported[stopping] = False

    def _advance_training_commands(self):
        if self._velocity_v2:
            return self._advance_curriculum_commands()
        self._command_steps -= 1
        self._retarget_steps -= 1
        ids = (self._command_steps <= 0).nonzero(as_tuple=False).flatten()
        if len(ids):
            self._standing_phase[ids] = ~self._standing_phase[ids]
            cfg = self.velocity_settings
            speed = cfg["minMovingSpeed"] + torch.rand(len(ids), device=self.device) * (
                cfg["maxSpeed"] - cfg["minMovingSpeed"])
            self.command_targets[ids, 0] = torch.where(self._standing_phase[ids], 0.0, speed)
            self._start_stop_timer(ids[self._standing_phase[ids]])
            # Starting another walk invalidates any unfinished stop measurement.
            walking = ids[~self._standing_phase[ids]]
            self._stop_elapsed[walking] = -1
            self._sample_duration(ids)
        retarget = (self._retarget_steps == 0) & ~self._standing_phase
        if retarget.any():
            cfg = self.velocity_settings
            self.command_targets[retarget, 0] = cfg["minMovingSpeed"] + torch.rand(
                int(retarget.sum()), device=self.device) * (cfg["maxSpeed"] - cfg["minMovingSpeed"])
            self._retarget_steps[retarget] = -1
        self.commands[:] = ramp_commands(self.commands, self.command_targets, self.dt, self.velocity_settings)

    def _advance_curriculum_commands(self):
        cfg = self.velocity_settings
        stage = cfg["curriculum"][self._curriculum_stage]
        self._command_steps -= 1
        reached = (self.commands[:, 0] - self.command_targets[:, 0]).abs() < 1e-5
        self._steady_steps[:] = torch.where(reached & ~self._standing_phase, self._steady_steps + 1, 0)
        due = self._command_steps <= 0
        ids = due.nonzero(as_tuple=False).flatten()
        if len(ids):
            self._standing_phase[ids] = ~self._standing_phase[ids] if not stage["continuousWalk"] else False
            self.command_targets[ids, 0] = torch.where(self._standing_phase[ids], 0.0, self._sample_moving_speed(len(ids)))
            self._start_stop_timer(ids[self._standing_phase[ids]])
            self._stop_elapsed[ids[~self._standing_phase[ids]]] = -1
            self._sample_duration(ids)
        retarget = (~due & ~self._standing_phase & ~self._retargeted
                    & (self._steady_steps * self.dt >= cfg["retargetSteadySeconds"]))
        if stage["retarget"] and retarget.any():
            self.command_targets[retarget, 0] = self._sample_moving_speed(int(retarget.sum()))
            self._retargeted[retarget] = True
            self._steady_steps[retarget] = 0
            # Leave at least two steady seconds after this new ramp as well.
            ramp_time = (self.command_targets[retarget, 0] - self.commands[retarget, 0]).abs() / min(cfg["acceleration"], cfg["deceleration"])
            remaining = torch.ceil((ramp_time + cfg["retargetSteadySeconds"]) / self.dt).long()
            self._command_steps[retarget] = torch.maximum(self._command_steps[retarget], remaining)
        self.commands[:] = ramp_commands(self.commands, self.command_targets, self.dt, cfg)

    def set_velocity_target(self, speed):
        """Select external replay/joystick control; call prepare_velocity_command once per action."""
        self._external_velocity_control = True
        speed = torch.as_tensor(speed, device=self.device, dtype=torch.float32).reshape(-1)
        speed = speed.clamp(0, self.velocity_settings["maxSpeed"])
        speed = speed * (speed >= self.velocity_settings["deadzone"])
        if speed.numel() not in (1, self.num_envs):
            raise ValueError("Expected one velocity target or one per environment")
        stopping = (self.command_targets[:, 0] > 0) & (speed == 0)
        self._start_stop_timer(stopping)
        self._stop_elapsed[speed.expand(self.num_envs) > 0] = -1
        self.command_targets[:, 0] = speed

    def prepare_velocity_command(self):
        if not self._external_velocity_control:
            raise RuntimeError("Set an external velocity target before preparing its observation")
        self.commands[:] = ramp_commands(self.commands, self.command_targets, self.dt, self.velocity_settings)
        self._write_commands()
        return self.obs_buf

    def pre_physics_step(self, actions):
        # These are the commands the actor saw; the scheduler updates NEXT obs only.
        self._reward_commands.copy_(self.commands)
        self._reward_targets.copy_(self.command_targets)
        self._previous_action.copy_(self.actions)
        if self._use_encoder and not self._velocity_v2:
            self._encoder_mask[:, 0] = encoder_command_mask(self.commands, self.velocity_settings)
            self._encoder_mask *= (self.progress_buf >= self._num_amp_obs_steps - 1).unsqueeze(-1)
        super().pre_physics_step(actions)

    def _motion_state(self):
        heading = torch_utils.calc_heading_quat_inv(self._humanoid_root_states[:, 3:7])
        velocity = quat_rotate(heading, self._humanoid_root_states[:, 7:10])
        omega = quat_rotate(heading, self._humanoid_root_states[:, 10:13])
        gravity = torch.zeros_like(velocity)
        gravity[:, 2] = -1
        gravity = quat_rotate_inverse(self._rigid_body_rot[:, 0], gravity)
        tilt = torch.acos((-gravity[:, 2]).clamp(-1, 1))
        feet = self._rigid_body_vel[:, self._feet_ids, :2].norm(dim=-1)
        return velocity, omega[:, 2], tilt, feet

    def _compute_reward(self, actions):
        velocity, yaw_rate, tilt, feet = self._motion_state()
        posture = None
        if self._velocity_v2:
            position = self._humanoid_root_states[:, :3]
            delta = position - self._progress_positions[self._progress_cursor]
            heading = torch_utils.calc_heading_quat_inv(self._humanoid_root_states[:, 3:7])
            self._forward_progress_speed[:] = quat_rotate(heading, delta)[:, 0] / (self._progress_positions.shape[0] * self.dt)
            self._progress_positions[self._progress_cursor] = position
            self._progress_cursor = (self._progress_cursor + 1) % self._progress_positions.shape[0]
            if self._use_encoder:
                valid = self.progress_buf >= max(self._num_amp_obs_steps - 1, self._progress_positions.shape[0])
                self._encoder_mask[:, 0] = encoder_motion_mask(self._reward_commands, self._forward_progress_speed, valid, self.velocity_settings)
            posture = self._posture_state(actions)
        reward, components = task_rewards(
            velocity, yaw_rate, tilt, self._dof_vel[:, self._leg_ids], feet,
            actions, self._previous_action, self.dof_force_tensor, self.motor_efforts,
            self._dof_pos, self.dof_limits_lower, self.dof_limits_upper,
            self._reward_commands, self.velocity_settings, posture)
        self.rew_buf[:] = reward
        self.extras["reward_components"] = {key: value.detach() for key, value in components.items()}

    def _posture_state(self, actions):
        cfg = self.velocity_settings
        foot_yaw, foot_tilt = foot_geometry(self._humanoid_root_states[:, 3:7],
                                          self._rigid_body_rot[:, self._feet_ids], cfg["footForwardAxis"])
        stance = ((self._contact_forces[:, self._feet_ids, 2] > cfg["stanceContactForce"])
                  & (self._rigid_body_pos[:, self._feet_ids, 2] < cfg["stanceMaxHeight"]))
        error = self._dof_pos - self._stand_pose
        pose_cfg = cfg["standPose"]
        pose_error = ((error[:, self._leg_ids] / pose_cfg["legSigma"]).square().mean(-1)
                      + pose_cfg["upperWeight"] * (error[:, self._upper_ids] / pose_cfg["upperSigma"]).square().mean(-1))
        return dict(foot_yaw=foot_yaw, foot_tilt=foot_tilt, stance=stance,
                    hip_yaw=self._dof_pos[:, self._hip_yaw_ids], stand_pose_error=pose_error,
                    pd_targets=self._raw_action_to_pd_targets(actions), knee_ids=self._knee_ids)

    def _compute_reset(self):
        super()._compute_reset()
        velocity, yaw_rate, _, feet = self._motion_state()
        speed = velocity[:, :2].norm(dim=-1)
        leg_speed = self._dof_vel[:, self._leg_ids].square().mean(-1).sqrt()
        quiet_command = self._reward_commands[:, 0] <= stand_command_threshold(self.velocity_settings)
        metrics = self.extras["motion_metrics"]
        metrics.update({
            "target_vx_m_s": self._reward_targets[:, 0],
            "command_vx_m_s": self._reward_commands[:, 0],
            "vx_m_s": velocity[:, 0], "vy_abs_m_s": velocity[:, 1].abs(),
            "yaw_rate_abs_rad_s": yaw_rate.abs(),
            "vx_error_m_s": (velocity[:, 0] - self._reward_commands[:, 0]).abs(),
            "stand_speed_m_s": speed[quiet_command],
            "stand_foot_speed_m_s": feet.mean(-1)[quiet_command],
            "stand_leg_speed_rad_s": leg_speed[quiet_command],
        })
        if self._use_encoder:
            metrics["encoder_mask"] = self._encoder_mask[:, 0]
        if self._velocity_v2:
            posture = self._posture_state(self.actions)
            targets = posture["pd_targets"]
            metrics.update({
                "forward_progress_m_s": self._forward_progress_speed,
                "foot_heading_abs_deg": torch.rad2deg(posture["foot_yaw"].abs()).flatten(),
                "hip_yaw_abs_deg": torch.rad2deg(posture["hip_yaw"].abs()).flatten(),
                "stance_foot_tilt_deg": torch.rad2deg(posture["foot_tilt"])[posture["stance"]],
                "pd_target_outside_fraction": ((targets < self.dof_limits_lower) | (targets > self.dof_limits_upper)).float().mean(-1),
                "knee_raw_target_below_limit_fraction": (targets[:, self._knee_ids] < self.dof_limits_lower[self._knee_ids]).float().mean(-1),
                "curriculum_stage": torch.full_like(speed, ord(self._curriculum_stage) - ord("A")),
            })
        error = (velocity[:, 0] - self._reward_commands[:, 0]).abs()
        for name, low, high in (("low", .15, .4), ("mid", .4, .65), ("high", .65, float("inf"))):
            moving = (self._reward_commands[:, 0] >= low) & (self._reward_commands[:, 0] < high)
            metrics["vx_error_" + name + "_m_s"] = error[moving]
        active = (self._stop_elapsed >= 0) & ~self._stop_reported
        self._stop_elapsed[active] += self.dt
        quiet = stop_quiet(self._reward_commands[:, 0], speed, feet.max(-1).values, leg_speed, self.velocity_settings)
        self._stop_hold[:] = torch.where(active & quiet, self._stop_hold + self.dt, 0.0)
        settled = active & (self._stop_hold >= .5) & (self._terminate_buf == 0)
        self.extras["event_metrics"] = {"stop_time_s": self._stop_elapsed[settled].clone()}
        self._stop_reported[settled] = True

    def post_physics_step(self):
        super().post_physics_step()
        if self._use_encoder:
            self.extras["enc_reward_mask"] = self._encoder_mask.clone()
        if not self._external_velocity_control:
            self._advance_training_commands()
            self._write_commands()

    def get_policy_config(self):
        return policy_spec(self.velocity_settings, self.dt, self.cfg["env"]["motion_file"], self._algorithm)

    def validate_policy_config(self, saved):
        expected = self.get_policy_config()
        if saved != expected:
            raise ValueError("Velocity task/checkpoint schema differs; start a new training run")

    def velocity_telemetry(self):
        velocity, yaw_rate, _, feet = self._motion_state()
        columns = [self._reward_targets[:, 0], self._reward_commands[:, 0],
                              velocity[:, 0], velocity[:, 1], yaw_rate,
                              (velocity[:, 0] - self._reward_commands[:, 0]).abs()]
        if self._use_encoder:
            columns.append(self._encoder_mask[:, 0])
        columns.extend([feet.mean(-1), self._dof_vel[:, self._leg_ids].square().mean(-1).sqrt(), self.rew_buf])
        values = torch.stack(columns, dim=-1)
        return values.detach().cpu().numpy()

    def velocity_telemetry_columns(self):
        columns = ["target_vx", "command_vx", "heading_vx", "heading_vy", "heading_yaw_rate", "vx_error"]
        if self._use_encoder:
            columns.append("encoder_mask")
        return columns + ["foot_speed", "leg_joint_speed_rms", "task_reward"]

    def velocity_posture_telemetry(self):
        """Extra diagnostics are separate so existing stop-summary columns stay stable."""
        if not self._velocity_v2:
            return {}
        posture = self._posture_state(self.actions)
        values = {"forward_progress_m_s": self._forward_progress_speed,
                  "foot_speed_max": self._motion_state()[3].max(-1).values}
        for i, side in enumerate(("left", "right")):
            values[side + "_foot_yaw_deg"] = torch.rad2deg(posture["foot_yaw"][:, i])
            values[side + "_foot_tilt_deg"] = torch.rad2deg(posture["foot_tilt"][:, i])
            values[side + "_foot_stance"] = posture["stance"][:, i]
        for i, name in enumerate(self._dof_names):
            values["pd_target_" + name] = posture["pd_targets"][:, i]
        executed = self._action_to_pd_targets(self.actions)
        for i in self._knee_ids:
            values["executed_pd_target_" + self._dof_names[i]] = executed[:, i]
        values["pd_target_outside_fraction"] = ((posture["pd_targets"] < self.dof_limits_lower)
                                                | (posture["pd_targets"] > self.dof_limits_upper)).float().mean(-1)
        return {name: value.detach().cpu().numpy() for name, value in values.items()}
