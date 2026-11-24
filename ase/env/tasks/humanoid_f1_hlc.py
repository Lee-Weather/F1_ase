"""F1 HLC speed task; underlying 29-action LLC physics remains unchanged."""
import math
from pathlib import Path

import torch
from isaacgym.torch_utils import quat_rotate, quat_rotate_inverse

from env.tasks.humanoid_f1 import HumanoidF1
from utils.f1_hlc import speed_task_reward
from utils.f1_hlc_gait import AlternatingGait, load_foot_vertices
from utils.f1_hlc_metrics import HLCMetrics
from utils.f1_hlc_curriculum import command_stage
from utils.f1_hlc_velocity import (VelocityMetrics, heading_velocity, sample_velocity_commands,
                                   standing_joint_groups, standing_pose_mse, velocity_task_reward)


class HumanoidF1HLC(HumanoidF1):
    def __init__(self, cfg, *args, **kwargs):
        self.hlc = cfg["env"]["hlc"]
        self.velocity_commands = self.hlc.get("commandMode") == "body_velocity"
        gait_cfg = self.hlc.get("gaitReward")
        if gait_cfg is not None and self.velocity_commands:
            raise ValueError("gaitReward requires scalar forward commands")
        sequence_enabled = any(stage.get("restartSequenceProbability", 0.)
                               for stage in self.hlc.get("commandCurriculum", []))
        self.commands_at_interval_boundary = self.velocity_commands or sequence_enabled or gait_cfg is not None
        self.training_epoch = 0
        if not 0 <= self.hlc.get("stopProbability", 0.) <= 1:
            raise ValueError("stopProbability must be in [0, 1]")
        super().__init__(cfg, *args, **kwargs)
        foot_names = ("left_ankle_roll_link", "right_ankle_roll_link")
        self._hlc_foot_ids = [self._body_names.index(n) for n in foot_names]
        self.gait = None
        if gait_cfg is not None:
            asset = cfg["env"]["asset"]
            mjcf = Path(asset["assetRoot"]) / asset["assetFileName"]
            vertices = load_foot_vertices(mjcf, foot_names)
            self.gait = AlternatingGait(self.num_envs, self.device, self.dt, gait_cfg, vertices)
        self._hlc_leg_ids = [i for i, n in enumerate(self._dof_names)
                             if any(part in n for part in ("hip", "knee", "ankle"))]
        self.command_next_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.restart_sequence_step = torch.full_like(self.command_next_step, 3) if sequence_enabled else None
        if self.velocity_commands:
            self.target_velocity = torch.zeros((self.num_envs, 3), device=self.device)
            self.command_groups = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            standing_pose = self.hlc.get("standingPoseWeight", 0.) > 0
            self._standing_groups = standing_joint_groups(self._dof_names) if standing_pose else None
            self._standing_pose_mse = None
            self.hlc_metrics = VelocityMetrics(self.device, standing_pose)
        else:
            self.target_speed = torch.ones(self.num_envs, device=self.device)
            self.target_direction = torch.zeros((self.num_envs, 3), device=self.device)
            self.target_direction[:, 0] = 1
            self.hlc_metrics = HLCMetrics(self.num_envs, self.device, self.dt, self.hlc)

    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)
        self._num_obs = 96
        self.cfg["env"]["numStates"] = 451

    def get_task_obs_size(self):
        return 3

    def _sample_commands(self, ids, reset=False):
        _, stage = command_stage(self.hlc, self.training_epoch)
        if self.velocity_commands:
            commands, groups = sample_velocity_commands(stage, len(ids), self.device)
            self.target_velocity[ids] = commands
            self.command_groups[ids] = groups
        else:
            self._sample_speed_commands(ids, stage, reset)
        steps = self.hlc["llcSteps"]
        lo, hi = [max(1, math.ceil(s/self.dt/steps)) for s in self.hlc["commandHoldSeconds"]]
        hold = torch.randint(lo, hi+1, (len(ids),), device=self.device)*steps
        self.command_next_step[ids] = hold if reset else self.progress_buf[ids]+hold

    def _sample_speed_commands(self, ids, stage, reset):
        low, high = stage["speedRange"]
        self.target_speed[ids] = low + (high-low)*torch.rand(len(ids), device=self.device)
        anchors = stage.get("speedAnchors", [])
        if anchors:
            selected = torch.rand(len(ids), device=self.device) < stage.get("anchorProbability", 0.)
            values = torch.tensor(anchors, dtype=self.target_speed.dtype, device=self.device)
            chosen = torch.randint(len(anchors), (len(ids),), device=self.device)
            self.target_speed[ids[selected]] = values[chosen[selected]]
        stop_probability = stage.get("stopProbability", 0.)
        if stop_probability or self.restart_sequence_step is not None:
            stop = torch.rand(len(ids), device=self.device) < stop_probability
            if self.restart_sequence_step is not None:
                stop = self._scheduled_stop_mask(ids, stage, reset, stop)
            self.target_speed[ids[stop]] = 0

    def _scheduled_stop_mask(self, ids, stage, reset, stop):
        if reset:
            probability = stage.get("restartSequenceProbability", 0.)
            scheduled = torch.rand(len(ids), device=self.device) < probability
            self.restart_sequence_step[ids] = torch.where(scheduled, 0, 3)
        step = self.restart_sequence_step[ids]
        self.restart_sequence_step[ids] = (step + 1).clamp(max=3)
        return (step == 1) | ((step == 3) & stop)

    def set_training_epoch(self, epoch):
        # Preserve an already-issued command until its normal expiry. This also
        # avoids changing the actor observation behind the PPO rollout buffer.
        self.training_epoch = int(epoch)

    def _reset_envs(self, env_ids):
        if len(env_ids):
            if self.gait is not None:
                self.gait.reset(env_ids)
            if not self.velocity_commands and hasattr(self, "hlc_metrics"):
                self.hlc_metrics.reset(env_ids)
            self._sample_commands(env_ids, reset=True)
        super()._reset_envs(env_ids)

    def pre_physics_step(self, actions):
        if not self.commands_at_interval_boundary:
            ids = (self.progress_buf >= self.command_next_step).nonzero(as_tuple=False).flatten()
            if len(ids):
                self._sample_commands(ids)
        super().pre_physics_step(actions)

    def advance_commands(self, active):
        """Issue the next command after the last transition of the old command."""
        ids = (active & (self.progress_buf >= self.command_next_step)).nonzero(as_tuple=False).flatten()
        if len(ids):
            self._sample_commands(ids)
            self._compute_observations(ids)
        return ids

    def _compute_observations(self, env_ids=None):
        super()._compute_observations(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        # Root orientation is available to an IMU-based actor. True linear
        # velocity remains confined to the critic and reward computation.
        if self.velocity_commands:
            command = self.target_velocity[ids]
        else:
            direction = quat_rotate_inverse(self._humanoid_root_states[ids, 3:7], self.target_direction[ids])
            command = torch.cat((direction[:, :2], self.target_speed[ids, None]), -1)
        self.obs_buf[ids, 93:] = command
        self.states_buf[ids, 448:] = command

    def _compute_reward(self, actions):
        if self.velocity_commands:
            foot_sq, leg_sq = self._standing_motion()
            if self._standing_groups is not None:
                self._standing_pose_mse = standing_pose_mse(
                    self._dof_pos, self._default_joint_pos, self._standing_groups)
            self.rew_buf[:] = velocity_task_reward(
                self._heading_velocity(), self.target_velocity, foot_sq, leg_sq, self.hlc, self._standing_pose_mse)
            return
        velocity = self._humanoid_root_states[:, 7:10]
        forward = torch.zeros_like(velocity)
        forward[:, 0] = 1
        facing = quat_rotate(self._humanoid_root_states[:, 3:7], forward)[:, 0]
        foot_speed_sq, leg_velocity_sq = self._standing_motion()
        self.rew_buf[:] = speed_task_reward(velocity, facing, self.target_speed,
                                            foot_speed_sq, leg_velocity_sq, self.hlc)

    def _heading_velocity(self):
        root = self._humanoid_root_states
        x, y, z, w = root[:, 3:7].unbind(dim=-1)
        fx, fy = 1 - 2 * (y.square() + z.square()), 2 * (x*y + z*w)
        return torch.stack(heading_velocity(fx, fy, root[:, 7], root[:, 8], root[:, 12]), dim=-1)

    def _standing_motion(self):
        return (self._rigid_body_vel[:, self._hlc_foot_ids].square().sum(-1).mean(-1),
                self._dof_vel[:, self._hlc_leg_ids].square().mean(-1))

    def _compute_reset(self):
        super()._compute_reset()
        if self.velocity_commands:
            stage, _ = command_stage(self.hlc, self.training_epoch)
            self.extras["motion_metrics"].update(
                task_reward=self.rew_buf.clone(),
                curriculum_stage=torch.full_like(self.rew_buf, stage))
            return
        foot_speed_sq, leg_velocity_sq = self._standing_motion()
        zero = self.target_speed <= 1e-6
        quiet = (self._humanoid_root_states[:, 7:9].norm(dim=-1) < self.hlc.get("quietPlanarSpeed", .05))
        quiet &= foot_speed_sq.sqrt() < self.hlc.get("quietFootSpeed", .1)
        quiet &= leg_velocity_sq.sqrt() < self.hlc.get("quietLegVelocity", .3)
        quiet &= ~self._terminate_buf.bool()
        self.extras["motion_metrics"].update(
            target_speed_m_s=self.target_speed.clone(),
            forward_speed_m_s=self._humanoid_root_states[:, 7].clone(),
            speed_error_abs_m_s=(self._humanoid_root_states[:, 7]-self.target_speed).abs(),
            lateral_speed_abs_m_s=self._humanoid_root_states[:, 8].abs(),
            task_reward=self.rew_buf.clone(),
            stop_command_fraction=zero.float(),
            # Both fractions use all samples; their ratio is quiet among zero commands.
            stop_quiet_fraction=(zero & quiet).float(),
            foot_speed_rms_m_s=foot_speed_sq.sqrt(),
            leg_velocity_rms_rad_s=leg_velocity_sq.sqrt())
        if self.hlc.get("commandCurriculum"):
            stage, _ = command_stage(self.hlc, self.training_epoch)
            self.extras["motion_metrics"]["curriculum_stage"] = torch.full_like(self.target_speed, stage)

    def observe_hlc_step(self, valid):
        foot_sq, leg_sq = self._standing_motion()
        failed = self.extras["terminate"].bool()
        if self.velocity_commands:
            actual = self._heading_velocity()
            quiet = (actual[:, :2].norm(dim=-1) < self.hlc["quietPlanarSpeed"])
            quiet &= actual[:, 2].abs() < self.hlc["quietYawRate"]
            quiet &= foot_sq.sqrt() < self.hlc["quietFootSpeed"]
            quiet &= leg_sq.sqrt() < self.hlc["quietLegVelocity"]
            self.hlc_metrics.step(self.command_groups, actual, self.target_velocity, valid, quiet & ~failed,
                                  failed, self._standing_pose_mse)
        else:
            self.hlc_metrics.step(self.target_speed, self._humanoid_root_states[:, 7],
                                  self._humanoid_root_states[:, 8], foot_sq.sqrt(), leg_sq.sqrt(),
                                  valid, failed, self.reset_buf.bool())
            if self.gait is not None:
                self._apply_gait_reward(valid, failed)

    def _apply_gait_reward(self, valid, failed):
        moving = self.target_speed > 1e-6
        live = valid & moving & ~failed
        feet = self._hlc_foot_ids
        recent, alternating, complete, slip, stance = self.gait.step(
            self._rigid_body_pos[:, feet], self._rigid_body_rot[:, feet], self._contact_forces[:, feet, 2], live)
        shaped = self.gait.moving_reward(self.rew_buf, recent, alternating, slip, stance)
        self.rew_buf[:] = torch.where(valid & moving, shaped, self.rew_buf)
        self.extras["motion_metrics"]["task_reward"] = self.rew_buf.clone()
        self.hlc_metrics.record_gait(live, recent, alternating, complete, slip, stance, self.gait.swing_increment)

    def get_training_metadata(self):
        result = super().get_training_metadata()
        result.update(task="HumanoidF1HLC", llc_actor_dim=93, latent_dim=32,
                      hlc=self.hlc.copy(), llc_control_hz=1/self.dt,
                      hlc_control_hz=1/(self.dt*self.hlc["llcSteps"]))
        return result
