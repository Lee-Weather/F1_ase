"""F1 LLC task: hardware actor observations and privileged training critic."""
import os
import xml.etree.ElementTree as ET

import torch
from isaacgym.torch_utils import quat_rotate, quat_rotate_inverse, quat_mul

from env.tasks.humanoid import compute_humanoid_observations_max
from env.tasks.humanoid_amp import HumanoidAMP
from env.tasks.humanoid_view_motion import HumanoidViewMotion
from utils import torch_utils
from utils.f1_motion_lib import F1MotionLib


class HumanoidF1(HumanoidAMP):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        self._initial_dof_pos[:] = self._default_joint_pos
        self.actions = torch.zeros((self.num_envs, 29), device=self.device)
        self._critic_body_ids = self._build_key_body_ids_tensor(self._critic_body_names)

    def _setup_character_props(self, key_bodies):
        self._num_actions, self._num_obs, self._dof_obs_size = 29, 93, 29
        self._dof_offsets = list(range(30))
        self.cfg["env"]["numStates"] = 448
        self._num_amp_obs_per_step = 13 + 29 + 29 + 3 * len(key_bodies)

    def _configure_asset(self, asset):
        self._dof_names = self.gym.get_asset_dof_names(asset)
        self._body_names = self.gym.get_asset_rigid_body_names(asset)
        urdf = os.path.join(self.cfg["env"]["asset"]["assetRoot"], "f1/urdf/f1_29dof.urdf")
        joints = {j.get("name"): j for j in ET.parse(urdf).getroot().findall("joint") if j.get("type") == "revolute"}
        assert len(self._dof_names) == 29 and len(self._body_names) == 53
        assert set(self._dof_names) == set(joints)
        self._critic_body_names = ["base_link"] + [joints[n].find("child").get("link") for n in self._dof_names]
        self._dof_body_ids = [self._body_names.index(n) for n in self._critic_body_names[1:]]
        self._default_joint_pos = torch.tensor([self.cfg["env"]["defaultJointPos"][n] for n in self._dof_names], dtype=torch.float32, device=self.device)
        props = self.gym.get_asset_dof_properties(asset)
        for i, name in enumerate(self._dof_names):
            group = next(g for g in self.cfg["env"]["pdGains"] if g in name)
            props["stiffness"][i], props["damping"][i] = self.cfg["env"]["pdGains"][group]
            limit = joints[name].find("limit")
            props["effort"][i], props["velocity"][i] = float(limit.get("effort")), float(limit.get("velocity"))
            if "armature" in props.dtype.names:
                props["armature"][i] = self.cfg["env"]["armature"]
        self._f1_dof_props = props

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        from isaacgym import gymapi
        props = self._f1_dof_props.copy()
        props["driveMode"] = gymapi.DOF_MODE_POS if self._pd_control else gymapi.DOF_MODE_NONE
        self.gym.set_actor_dof_properties(env_ptr, self.humanoid_handles[-1], props)
        # Imported convex leg meshes overlap within a kinematic chain. Shared
        # bits exclude those pairs; disjoint bits preserve left/right collisions.
        shapes = self.gym.get_actor_rigid_shape_properties(env_ptr, self.humanoid_handles[-1])
        ranges = self.gym.get_actor_rigid_body_shape_indices(env_ptr, self.humanoid_handles[-1])
        for group_id, names in enumerate(self.cfg["env"].get("selfCollisionFilterGroups", [])):
            for name in names:
                indices = ranges[self._body_names.index(name)]
                for shape_id in range(indices.start, indices.start + indices.count):
                    shapes[shape_id].filter |= 1 << group_id
        self.gym.set_actor_rigid_shape_properties(env_ptr, self.humanoid_handles[-1], shapes)

    def _create_envs(self, *args):
        super()._create_envs(*args)
        props = self.gym.get_actor_dof_properties(self.envs[0], self.humanoid_handles[0])
        self.motor_efforts = torch.tensor(props["effort"].copy(), device=self.device)
        self.max_motor_effort = float(self.motor_efforts.max())
        assert self._body_names[0] == "base_link"

    def _build_pd_action_offset_scale(self):
        self._pd_action_offset = self._default_joint_pos.clone()
        self._pd_action_scale = torch.full_like(self._pd_action_offset, self.cfg["env"]["actionScale"])

    def _build_termination_heights(self):
        # F1 termination uses the base height, tilt and disallowed contacts below.
        self._termination_heights = torch.zeros(self.num_bodies, device=self.device)

    def _load_motion(self, motion_file):
        self._motion_lib = F1MotionLib(motion_file, self._dof_names, self._body_names, self._key_body_ids, self.device)

    def _reset_envs(self, env_ids):
        if len(env_ids) == 0:
            return
        self.actions[env_ids] = 0
        super()._reset_envs(env_ids)

    def _compute_observations(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        rot = self._rigid_body_rot[ids, 0]
        gravity = torch.zeros_like(self._rigid_body_ang_vel[ids, 0])
        gravity[:, 2] = -1
        obs = torch.cat((quat_rotate_inverse(rot, self._rigid_body_ang_vel[ids, 0]),
                         quat_rotate_inverse(rot, gravity),
                         self._dof_pos[ids] - self._default_joint_pos,
                         self._dof_vel[ids], self.actions[ids]), dim=-1)
        self.obs_buf[ids] = obs
        bodies = self._critic_body_ids
        self.states_buf[ids] = compute_humanoid_observations_max(
            self._rigid_body_pos[ids][:, bodies], self._rigid_body_rot[ids][:, bodies],
            self._rigid_body_vel[ids][:, bodies], self._rigid_body_ang_vel[ids][:, bodies], False, True)

    def _compute_reset(self):
        gravity = torch.zeros((self.num_envs, 3), device=self.device)
        gravity[:, 2] = -1
        local_gravity = quat_rotate_inverse(self._rigid_body_rot[:, 0], gravity)
        tilt = torch.acos(torch.clamp(-local_gravity[:, 2], -1, 1))
        contact = self._contact_forces.clone()
        contact[:, self._contact_body_ids] = 0
        # Net forces also contain self contacts; a fall contact must be near
        # the floor. Base height and tilt still terminate independently.
        bad_contact = ((contact.norm(dim=-1) > self.cfg["env"]["terminationContactForce"]) &
                       (self._rigid_body_pos[:, :, 2] < self.cfg["env"]["terminationContactHeight"])).any(dim=-1)
        base_low = self._rigid_body_pos[:, 0, 2] < self.cfg["env"]["terminationHeight"]
        base_tilt = tilt > self.cfg["env"]["terminationTilt"]
        eligible = (self.progress_buf > 1) & self._enable_early_termination
        fallen = (base_low | base_tilt | bad_contact) & eligible
        time_out = self.progress_buf >= self.max_episode_length - 1
        self._terminate_buf[:] = fallen.long()
        self.reset_buf[:] = (fallen | time_out).long()
        # Telemetry only; one episode can have multiple simultaneous reasons.
        self.extras["termination_reasons"] = {
            "time_out": time_out,
            "base_height": base_low & eligible,
            "base_tilt": base_tilt & eligible,
            "illegal_ground_contact": bad_contact & eligible,
        }

    def _build_amp_observations(self, root_pos, root_rot, root_vel, root_ang_vel,
                                dof_pos, dof_vel, key_body_pos, *unused):
        return build_f1_amp_observations(root_pos, root_rot, root_vel, root_ang_vel,
                                         dof_pos, dof_vel, key_body_pos)


@torch.jit.script
def build_f1_amp_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos):
    heading = torch_utils.calc_heading_quat_inv(root_rot)
    rot = torch_utils.quat_to_tan_norm(quat_mul(heading, root_rot))
    keys = key_body_pos - root_pos.unsqueeze(1)
    h = heading.unsqueeze(1).expand(-1, keys.shape[1], -1).reshape(-1, 4)
    keys = quat_rotate(h, keys.reshape(-1, 3)).reshape(root_pos.shape[0], key_body_pos.shape[1] * 3)
    return torch.cat((root_pos[:, 2:3], rot, quat_rotate(heading, root_vel),
                      quat_rotate(heading, root_ang_vel), dof_pos, dof_vel, keys), dim=-1)


class HumanoidF1ViewMotion(HumanoidViewMotion, HumanoidF1):
    """Use the existing kinematic viewer with F1 assets, observations and loader."""
    pass
