"""Velocity-conditioned ASE task with hardware-compatible actor observations."""
import torch
from isaacgym.torch_utils import quat_rotate, quat_rotate_inverse

from env.tasks.humanoid_f1 import HumanoidF1
from utils import torch_utils
from utils.f1_velocity import (encoder_command_mask, policy_spec, ramp_commands,
                               task_rewards, velocity_settings)


class HumanoidF1Velocity(HumanoidF1):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.velocity_settings = velocity_settings(cfg["env"])
        self._external_velocity_control = False
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        self.commands = torch.zeros((self.num_envs, 3), device=self.device)
        self.command_targets = torch.zeros_like(self.commands)
        self._reward_commands = torch.zeros_like(self.commands)
        self._reward_targets = torch.zeros_like(self.commands)
        self._previous_action = torch.zeros_like(self.actions)
        self._encoder_mask = torch.zeros((self.num_envs, 1), device=self.device)
        self._standing_phase = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._command_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._retarget_steps = torch.full_like(self._command_steps, -1)
        self._leg_ids = torch.tensor([i for i, name in enumerate(self._dof_names)
                                     if any(part in name for part in ("hip", "knee", "ankle"))], device=self.device)
        self._feet_ids = self._build_key_body_ids_tensor(["left_ankle_roll_link", "right_ankle_roll_link"])
        self._stop_elapsed = torch.full((self.num_envs,), -1.0, device=self.device)
        self._stop_hold = torch.zeros(self.num_envs, device=self.device)
        self._stop_reported = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

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

    def _reset_envs(self, env_ids):
        if len(env_ids) == 0:
            return
        self.commands[env_ids] = 0
        self.command_targets[env_ids] = 0
        self._reward_commands[env_ids] = 0
        self._reward_targets[env_ids] = 0
        self._previous_action[env_ids] = 0
        self._encoder_mask[env_ids] = 0
        self._standing_phase[env_ids] = True
        self._stop_elapsed[env_ids] = -1
        self._stop_hold[env_ids] = 0
        self._stop_reported[env_ids] = False
        self._sample_duration(env_ids)
        super()._reset_envs(env_ids)

    def _start_stop_timer(self, stopping):
        self._stop_elapsed[stopping] = 0
        self._stop_hold[stopping] = 0
        self._stop_reported[stopping] = False

    def _advance_training_commands(self):
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
        reward, components = task_rewards(
            velocity, yaw_rate, tilt, self._dof_vel[:, self._leg_ids], feet,
            actions, self._previous_action, self.dof_force_tensor, self.motor_efforts,
            self._dof_pos, self.dof_limits_lower, self.dof_limits_upper,
            self._reward_commands, self.velocity_settings)
        self.rew_buf[:] = reward
        self.extras["reward_components"] = {key: value.detach() for key, value in components.items()}

    def _compute_reset(self):
        super()._compute_reset()
        velocity, yaw_rate, _, feet = self._motion_state()
        speed = velocity[:, :2].norm(dim=-1)
        leg_speed = self._dof_vel[:, self._leg_ids].square().mean(-1).sqrt()
        quiet_command = self._reward_commands[:, 0] <= self.velocity_settings["encoderZeroSpeed"]
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
            "encoder_mask": self._encoder_mask[:, 0],
        })
        error = (velocity[:, 0] - self._reward_commands[:, 0]).abs()
        for name, low, high in (("low", .15, .4), ("mid", .4, .65), ("high", .65, float("inf"))):
            moving = (self._reward_commands[:, 0] >= low) & (self._reward_commands[:, 0] < high)
            metrics["vx_error_" + name + "_m_s"] = error[moving]
        active = (self._stop_elapsed >= 0) & ~self._stop_reported
        self._stop_elapsed[active] += self.dt
        quiet = quiet_command & (speed < .05) & (feet.max(-1).values < .1) & (leg_speed < .3)
        self._stop_hold[:] = torch.where(active & quiet, self._stop_hold + self.dt, 0.0)
        settled = active & (self._stop_hold >= .5) & (self._terminate_buf == 0)
        self.extras["event_metrics"] = {"stop_time_s": self._stop_elapsed[settled].clone()}
        self._stop_reported[settled] = True

    def post_physics_step(self):
        super().post_physics_step()
        self.extras["enc_reward_mask"] = self._encoder_mask.clone()
        if not self._external_velocity_control:
            self._advance_training_commands()
            self._write_commands()

    def get_policy_config(self):
        return policy_spec(self.velocity_settings, self.dt, self.cfg["env"]["motion_file"])

    def validate_policy_config(self, saved):
        expected = self.get_policy_config()
        if saved != expected:
            raise ValueError("Velocity task/checkpoint schema differs; start a new training run")

    def velocity_telemetry(self):
        velocity, yaw_rate, _, feet = self._motion_state()
        values = torch.stack((self._reward_targets[:, 0], self._reward_commands[:, 0],
                              velocity[:, 0], velocity[:, 1], yaw_rate,
                              (velocity[:, 0] - self._reward_commands[:, 0]).abs(),
                              self._encoder_mask[:, 0], feet.mean(-1),
                              self._dof_vel[:, self._leg_ids].square().mean(-1).sqrt(), self.rew_buf), dim=-1)
        return values.detach().cpu().numpy()
