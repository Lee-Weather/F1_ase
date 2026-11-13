"""F1 HLC speed task; underlying 29-action LLC physics remains unchanged."""
import math

import torch
from isaacgym.torch_utils import quat_rotate, quat_rotate_inverse

from env.tasks.humanoid_f1 import HumanoidF1


class HumanoidF1HLC(HumanoidF1):
    def __init__(self, cfg, *args, **kwargs):
        self.hlc = cfg["env"]["hlc"]
        super().__init__(cfg, *args, **kwargs)
        self.target_speed = torch.ones(self.num_envs, device=self.device)
        self.command_next_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.target_direction = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_direction[:, 0] = 1  # Fixed world heading, not a rotating body-forward goal.

    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)
        self._num_obs = 96
        self.cfg["env"]["numStates"] = 451

    def get_task_obs_size(self):
        return 3

    def _sample_commands(self, ids, reset=False):
        low, high = self.hlc["speedRange"]
        self.target_speed[ids] = low + (high-low)*torch.rand(len(ids), device=self.device)
        steps = self.hlc["llcSteps"]
        lo, hi = [max(1, math.ceil(s/self.dt/steps)) for s in self.hlc["commandHoldSeconds"]]
        hold = torch.randint(lo, hi+1, (len(ids),), device=self.device)*steps
        self.command_next_step[ids] = hold if reset else self.progress_buf[ids]+hold

    def _reset_envs(self, env_ids):
        if len(env_ids):
            self._sample_commands(env_ids, reset=True)
        super()._reset_envs(env_ids)

    def pre_physics_step(self, actions):
        ids = (self.progress_buf >= self.command_next_step).nonzero(as_tuple=False).flatten()
        if len(ids):
            self._sample_commands(ids)
        super().pre_physics_step(actions)

    def _compute_observations(self, env_ids=None):
        super()._compute_observations(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        # Root orientation is available to an IMU-based actor. True linear
        # velocity remains confined to the critic and reward computation.
        direction = quat_rotate_inverse(self._humanoid_root_states[ids, 3:7], self.target_direction[ids])
        command = torch.cat((direction[:, :2], self.target_speed[ids, None]), -1)
        self.obs_buf[ids, 93:] = command
        self.states_buf[ids, 448:] = command

    def _compute_reward(self, actions):
        velocity = self._humanoid_root_states[:, 7:10]
        forward = torch.zeros_like(velocity)
        forward[:, 0] = 1
        facing = quat_rotate(self._humanoid_root_states[:, 3:7], forward)[:, 0]
        error = velocity[:, 0]-self.target_speed
        self.rew_buf[:] = torch.exp(-self.hlc["speedTrackingScale"]*error.square()
                                   -self.hlc["lateralTrackingScale"]*velocity[:, 1].square()
                                   -self.hlc["headingTrackingScale"]*(1-facing))

    def _compute_reset(self):
        super()._compute_reset()
        self.extras["motion_metrics"].update(
            target_speed_m_s=self.target_speed.clone(),
            forward_speed_m_s=self._humanoid_root_states[:, 7].clone(),
            speed_error_abs_m_s=(self._humanoid_root_states[:, 7]-self.target_speed).abs(),
            lateral_speed_abs_m_s=self._humanoid_root_states[:, 8].abs(),
            task_reward=self.rew_buf.clone())

    def get_training_metadata(self):
        result = super().get_training_metadata()
        result.update(task="HumanoidF1HLC", llc_actor_dim=93, latent_dim=32,
                      hlc=self.hlc.copy(), llc_control_hz=1/self.dt,
                      hlc_control_hz=1/(self.dt*self.hlc["llcSteps"]))
        return result
