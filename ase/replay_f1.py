"""Replay an F1 AMP/ASE checkpoint in Isaac Gym or MuJoCo without training.

Example: python ase/replay_f1.py --engine mujoco --checkpoint MODEL --output DIR
Both engines use the same deterministic actor, latent seeds and initial states.
Outputs are one two-view video and one CSV per engine.
"""
import argparse
import csv
import itertools
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import yaml
from utils.f1_control import pd_action_parameters, action_to_targets, checkpoint_control
from utils.f1_velocity import load_env_config, scripted_speed

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "ase/data/assets/f1/mjcf/weichai_f1_29.xml"


def load_actor(checkpoint, device="cpu"):
    """Inference-only equivalent of ASEBuilder's F1 style actor and normalizer."""
    import torch
    import torch.nn.functional as F
    kwargs = {"weights_only": False} if "weights_only" in inspect.signature(torch.load).parameters else {}
    ckpt = torch.load(str(checkpoint), map_location=device, **kwargs)
    weights = {k[len("a2c_network."):]: v.detach().to(device) for k, v in ckpt["model"].items()}
    norm = ckpt["running_mean_std"]
    mean = norm["running_mean"].float().to(device)
    scale = torch.sqrt(norm["running_var"].float().to(device) + 1e-5)
    is_ase = "actor_mlp._style_dense.weight" in weights
    amp_layers = sorted(int(k.split(".")[1]) for k in weights
                        if k.startswith("actor_mlp.") and k.endswith(".weight") and k.split(".")[1].isdigit())

    def linear(x, name):
        return F.linear(x, weights[name + ".weight"], weights[name + ".bias"])

    def actor(obs, latent):
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
            obs = torch.clamp((obs - mean) / scale, -5, 5)
            if is_ase:
                style = F.relu(linear(latent, "actor_mlp._style_mlp.0"))
                style = F.relu(linear(style, "actor_mlp._style_mlp.2"))
                style = torch.tanh(linear(style, "actor_mlp._style_dense"))
                hidden = torch.cat((obs, style), dim=-1)
                for i in range(3):
                    hidden = F.relu(linear(hidden, "actor_mlp._dense_layers." + str(i)))
            else:
                hidden = obs
                for i in amp_layers:
                    hidden = F.relu(linear(hidden, "actor_mlp." + str(i)))
            return linear(hidden, "mu").clamp(-1, 1)

    print(json.dumps({"checkpoint_epoch": ckpt["epoch"], "frame": ckpt.get("frame"),
                      "actor_observations": int(mean.numel()), "deterministic": True}), flush=True)
    actor.epoch = int(ckpt["epoch"])
    actor.algorithm = "ASE" if is_ase else "AMP"
    actor.control_config = checkpoint_control(ckpt)
    actor.policy_config = ckpt.get("policy_config")
    actor.observation_size = int(mean.numel())
    return actor


def joint_names():
    return [j.get("name") for j in ET.parse(ASSET).iter("joint")]


def export_actor_npz(checkpoint, destination):
    """Export inference weights for hosts without a working PyTorch runtime."""
    import torch
    kwargs = {"weights_only": False} if "weights_only" in inspect.signature(torch.load).parameters else {}
    ckpt = torch.load(str(checkpoint), map_location="cpu", **kwargs)
    weights = {k[len("a2c_network."):]: v.detach().float().numpy()
               for k, v in ckpt["model"].items()
               if k.startswith("a2c_network.actor_mlp.") or k.startswith("a2c_network.mu.")}
    weights.update(mean=ckpt["running_mean_std"]["running_mean"].float().numpy(),
                   scale=np.sqrt(ckpt["running_mean_std"]["running_var"].float().numpy() + 1e-5),
                   epoch=np.array(ckpt["epoch"]),
                   policy_config=np.array(json.dumps(ckpt.get("policy_config"))),
                   control_config=np.array(json.dumps(checkpoint_control(ckpt))))
    np.savez_compressed(destination, **weights)


def load_actor_numpy(path):
    """Same normalized deterministic AMP/ASE actor using float32 NumPy operations."""
    with np.load(path, allow_pickle=False) as archive:
        weights = {k: archive[k] for k in archive.files}
    is_ase = "actor_mlp._style_dense.weight" in weights
    amp_layers = sorted(int(k.split(".")[1]) for k in weights
                        if k.startswith("actor_mlp.") and k.endswith(".weight") and k.split(".")[1].isdigit())

    def linear(x, name):
        return x @ weights[name + ".weight"].T + weights[name + ".bias"]

    def actor(obs, latent):
        hidden = np.clip((np.asarray(obs, dtype=np.float32) - weights["mean"]) / weights["scale"], -5, 5)
        if is_ase:
            style = np.maximum(linear(latent, "actor_mlp._style_mlp.0"), 0)
            style = np.maximum(linear(style, "actor_mlp._style_mlp.2"), 0)
            style = np.tanh(linear(style, "actor_mlp._style_dense"))
            hidden = np.concatenate((hidden, style), axis=-1)
            layers = ["actor_mlp._dense_layers." + str(i) for i in range(3)]
        else:
            layers = ["actor_mlp." + str(i) for i in amp_layers]
        for name in layers:
            hidden = np.maximum(linear(hidden, name), 0)
        return np.clip(linear(hidden, "mu"), -1, 1)

    actor.epoch = int(weights["epoch"])
    actor.algorithm = "ASE" if is_ase else "AMP"
    actor.control_config = checkpoint_control({"control_config": json.loads(str(weights["control_config"]))})
    actor.policy_config = json.loads(str(weights["policy_config"])) if "policy_config" in weights else None
    actor.observation_size = int(weights["mean"].size)
    return actor


def initial_case(index, cfg, names, motion_file=None):
    default = np.array([cfg["env"]["defaultJointPos"][n] for n in names])
    root = np.array([0., 0., cfg["env"]["initialHeight"], 1., 0., 0., 0.])  # wxyz
    state = dict(q=default, qd=np.zeros(29), root=root, velocity=np.zeros(6), init="default")
    if motion_file is not None and index > 0:
        with np.load(motion_file, allow_pickle=False) as d:
            frame = int(((index - 1) % 4) * (len(d["joint_pos"]) - 1) / 4)
            order = [d["joint_names"].tolist().index(n) for n in names]
            root = np.r_[d["body_pos_w"][frame, 0], d["body_quat_w"][frame, 0]]
            root[:2] = 0  # Recenter each reference state without changing its height or orientation.
            state = dict(q=d["joint_pos"][frame, order].copy(), qd=d["joint_vel"][frame, order].copy(),
                         root=root, velocity=np.r_[d["body_lin_vel_w"][frame, 0], d["body_ang_vel_w"][frame, 0]],
                         init=motion_file.name + ":frame" + str(frame))
    elif motion_file is None and 4 <= index < 8:
        folder = ROOT / "ase/data/motions/f1"
        entries = yaml.safe_load((folder / "dataset_f1.yaml").read_text())["motions"]
        motion = entries[(index - 4) * (len(entries) - 1) // 3]["file"]
        with np.load(folder / motion, allow_pickle=False) as d:
            order = [d["joint_names"].tolist().index(n) for n in names]
            state = dict(q=d["joint_pos"][0, order].copy(), qd=d["joint_vel"][0, order].copy(),
                         root=np.r_[d["body_pos_w"][0, 0], d["body_quat_w"][0, 0]],
                         velocity=np.r_[d["body_lin_vel_w"][0, 0], d["body_ang_vel_w"][0, 0]],
                         init=motion)
    return state


class Latents:
    def __init__(self, seed, switching):
        self.rng = np.random.RandomState(seed)
        self.switching, self.left, self.value = switching, 0, None

    def next(self):
        if self.left <= 0:
            self.value = self.rng.normal(size=32).astype(np.float32)
            self.value /= np.linalg.norm(self.value)
            self.left = int(self.rng.randint(1, 150)) if self.switching else 1000000
        self.left -= 1
        return self.value


class Video:
    def __init__(self, path):
        import shutil
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        encoder_env = os.environ.copy()
        encoder_env.pop("LD_LIBRARY_PATH", None)  # System ffmpeg must use system libraries.
        self.proc = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", "1280x560", "-r", "25", "-i", "-", "-an",
            "-c:v", "libx264", "-crf", "21", "-preset", "fast", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE, env=encoder_env)
        from PIL import ImageFont
        try:
            self.font = ImageFont.truetype("arial.ttf" if os.name == "nt" else "DejaVuSans.ttf", 20)
        except OSError:
            self.font = ImageFont.load_default()

    def write(self, views, text):
        from PIL import Image, ImageDraw
        frame = Image.new("RGB", (1280, 560), (22, 26, 33))
        for i, view in enumerate(views):
            frame.paste(Image.fromarray(view[:, :, :3].copy()), (640 * i, 80))
        ImageDraw.Draw(frame).text((18, 12), text, font=self.font, fill="white")
        self.proc.stdin.write(np.asarray(frame).tobytes())

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("Video encoding failed")


class GymReplay:
    def __init__(self, cfg, names, velocity=False, record_video=True, viewer=False):
        from isaacgym import gymapi
        import torch
        from env.tasks.humanoid_f1 import HumanoidF1
        if velocity:
            from env.tasks.humanoid_f1_velocity import HumanoidF1Velocity
        params = gymapi.SimParams()
        params.dt, params.substeps = cfg["sim"]["dt"], cfg["sim"]["substeps"]
        params.use_gpu_pipeline = params.physx.use_gpu = True
        for key, value in cfg["sim"]["physx"].items():
            if hasattr(params.physx, key):
                setattr(params.physx, key, value)
        cfg["env"].update(numEnvs=1, stateInit="Default", enableCameraSensors=record_video)
        motion = "dataset_f1_walk_stop.yaml" if velocity else "dataset_f1.yaml"
        cfg["env"]["motion_file"] = str(ROOT / "ase/data/motions/f1" / motion)
        cfg["env"]["asset"]["assetRoot"] = str(ROOT / "ase/data/assets")
        task_class = HumanoidF1Velocity if velocity else HumanoidF1
        self.task = task_class(cfg, params, gymapi.SIM_PHYSX, "cuda", 0, not viewer)
        if velocity:
            self.task.set_velocity_target(0.0)
        assert list(self.task._dof_names) == names, "Actor joint order differs from asset"
        self.gym, self.sim, self.env = self.task.gym, self.task.sim, self.task.envs[0]
        properties = gymapi.CameraProperties()
        properties.width, properties.height = 640, 480
        self.cameras = [self.gym.create_camera_sensor(self.env, properties) for _ in range(2)] if record_video else []
        if self.cameras and min(self.cameras) < 0:
            raise RuntimeError("Isaac Gym camera creation failed")
        self.device, self.dt = self.task.device, self.task.dt

    def reset(self, state):
        import torch
        task = self.task
        task._initial_dof_pos[:] = torch.as_tensor(state["q"], device=self.device)
        task._initial_dof_vel[:] = torch.as_tensor(state["qd"], device=self.device)
        root = state["root"]
        task._initial_humanoid_root_states[0] = torch.as_tensor(
            np.r_[root[:3], root[4:7], root[3], state["velocity"]], device=self.device)
        task.reset()
        self.steps = 0
        return task.obs_buf.clone()

    def command_observation(self, speed):
        self.task.set_velocity_target(speed)
        return self.task.prepare_velocity_command().clone()

    def step(self, action):
        task = self.task
        task.step(action)
        self.gym.fetch_results(self.sim, True)
        self.steps += 1
        root = task._humanoid_root_states[0].detach().cpu().numpy()
        rot = root[[6, 3, 4, 5]]
        values = kinematics(root[:3], rot, root[7:10], root[10:13])
        reasons = [k for k, v in task.extras["termination_reasons"].items() if bool(v[0])]
        return task.obs_buf.clone(), values, reasons, task._dof_pos[0].detach().cpu().numpy()

    def render(self):
        from isaacgym import gymapi
        pos = self.task._humanoid_root_states[0, :3].cpu().numpy()
        for camera, offset in zip(self.cameras, ((2., -2.3, 1.), (0., -3., .6))):
            self.gym.set_camera_location(camera, self.env, gymapi.Vec3(*(pos + offset)),
                                         gymapi.Vec3(pos[0], pos[1], max(.4, pos[2])))
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        return [self.gym.get_camera_image(self.sim, self.env, cam, gymapi.IMAGE_COLOR)
                .reshape(480, 640, 4) for cam in self.cameras]

    def close(self):
        for cam in self.cameras:
            self.gym.destroy_camera_sensor(self.sim, self.env, cam)
        if self.task.viewer is not None:
            self.gym.destroy_viewer(self.task.viewer)
        self.gym.destroy_sim(self.sim)


class JoystickSpeed:
    """Optional pygame input, available only when explicitly requested."""
    def __init__(self, index, axis, sign, stop_button, maximum):
        os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        if os.name != "nt" and not os.environ.get("DISPLAY"):
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        try:
            import pygame
        except ImportError as error:
            raise RuntimeError("Joystick mode requires the optional pygame>=2 package") from error
        self.pg = pygame
        pygame.display.init()
        pygame.display.set_mode((1, 1), pygame.HIDDEN)
        pygame.joystick.init()
        if not 0 <= index < pygame.joystick.get_count():
            pygame.quit()
            raise ValueError("Requested joystick was not found")
        self.pad = pygame.joystick.Joystick(index)
        if not 0 <= axis < self.pad.get_numaxes() or not -1 <= stop_button < self.pad.get_numbuttons():
            pygame.quit()
            raise ValueError("Joystick axis or stop button is outside the device range")
        self.axis, self.sign, self.stop_button, self.maximum = axis, sign, stop_button, maximum
        self.attached = True

    def read(self):
        for event in self.pg.event.get():
            if event.type == self.pg.JOYDEVICEREMOVED and event.instance_id == self.pad.get_instance_id():
                self.attached = False
                print("Joystick disconnected: target speed is zero", flush=True)
        if not self.attached or (self.stop_button >= 0 and self.pad.get_button(self.stop_button)):
            return 0.0
        return float(np.clip(self.sign * self.pad.get_axis(self.axis), 0, 1) * self.maximum)

    def close(self):
        self.pg.quit()


def rotation(quat):
    w, x, y, z = quat
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def kinematics(pos, quat, linear, angular):
    rot = rotation(quat)
    vel, omega = rot.T @ linear, rot.T @ angular
    tilt = np.arccos(np.clip(rot[2, 2], -1, 1))
    return [*pos, vel[0], vel[1], omega[2], tilt]


class MujocoReplay:
    def __init__(self, cfg, names):
        import mujoco
        self.mj, self.cfg = mujoco, cfg["env"]
        xml = ET.parse(ASSET).getroot()
        for mesh in xml.findall(".//asset/mesh"):
            mesh.set("file", str((ASSET.parent / mesh.get("file")).resolve()))
        ET.SubElement(xml, "option", timestep="0.005", gravity="0 0 -9.81", integrator="implicitfast")
        assets = xml.find("asset")
        ET.SubElement(assets, "texture", name="replay_grid", type="2d", builtin="checker",
                      rgb1="0.2 0.23 0.28", rgb2="0.3 0.33 0.38", width="256", height="256")
        ET.SubElement(assets, "material", name="replay_floor", texture="replay_grid", texrepeat="10 10")
        ET.SubElement(xml.find("worldbody"), "geom", name="replay_ground", type="plane",
                      size="100 100 0.1", material="replay_floor", friction="0.8 0.02 0.001", group="2")
        ET.SubElement(xml.find("worldbody"), "light", pos="0 0 5", dir="0 0 -1", directional="true")
        contact = xml.find("contact")
        if contact is None:
            contact = ET.SubElement(xml, "contact")
        for group in self.cfg["selfCollisionFilterGroups"]:
            for a, b in itertools.combinations(group, 2):
                ET.SubElement(contact, "exclude", body1=a, body2=b)
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
        model = self.model
        model.vis.global_.offwidth, model.vis.global_.offheight = 640, 480
        self.data = mujoco.MjData(model)
        self.jids = np.array([model.joint(n).id for n in names])
        self.qids, self.vids = model.jnt_qposadr[self.jids], model.jnt_dofadr[self.jids]
        self.aids = np.array([np.flatnonzero(model.actuator_trnid[:, 0] == jid)[0] for jid in self.jids])
        self.default = np.array([self.cfg["defaultJointPos"][n] for n in names])
        gains = [next(v for k, v in self.cfg["pdGains"].items() if k in n) for n in names]
        self.kp, self.kd = np.array(gains).T
        urdf = ET.parse(ROOT / "ase/data/assets/f1/urdf/f1_29dof.urdf").getroot()
        limits = {j.get("name"): j.find("limit")
                  for j in urdf.findall("joint") if j.get("type") == "revolute"}
        self.effort = np.array([float(limits[n].get("effort")) for n in names])
        self.action_offset, self.action_negative, self.action_positive = pd_action_parameters(
            self.cfg, names, [float(limits[n].get("lower")) for n in names],
            [float(limits[n].get("upper")) for n in names])
        model.dof_damping[self.vids] = 0
        model.dof_armature[self.vids] = self.cfg["armature"]
        self.base = model.body("base_link").id
        self.floor = model.geom("replay_ground").id
        self.allowed = {model.body(n).id for n in self.cfg["contactBodies"]}
        self.renderer = mujoco.Renderer(model, height=480, width=640)
        self.camera = mujoco.MjvCamera()
        self.option = mujoco.MjvOption()
        self.option.geomgroup[0], self.option.geomgroup[1], self.option.geomgroup[2] = 0, 1, 1
        self.device, self.dt = "cpu", .02

    def obs(self):
        rot = self.data.xmat[self.base].reshape(3, 3)
        velocity = np.zeros(6)
        self.mj.mj_objectVelocity(self.model, self.data, self.mj.mjtObj.mjOBJ_BODY, self.base, velocity, 0)
        obs = np.r_[rot.T @ velocity[:3], rot.T @ [0., 0., -1.],
                    self.data.qpos[self.qids] - self.default, self.data.qvel[self.vids], self.action]
        return obs.astype(np.float32)[None, :]

    def reset(self, state):
        self.mj.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = state["root"]
        self.data.qpos[self.qids], self.data.qvel[self.vids] = state["q"], state["qd"]
        self.data.qvel[:3] = state["velocity"][:3]
        self.data.qvel[3:6] = rotation(state["root"][3:7]).T @ state["velocity"][3:6]
        self.action, self.steps = np.zeros(29), 0
        self.mj.mj_forward(self.model, self.data)
        return self.obs()

    def step(self, action):
        self.action = action[0].copy() if isinstance(action, np.ndarray) else action[0].cpu().numpy()
        target = action_to_targets(self.action, self.action_offset, self.action_negative, self.action_positive)
        for _ in range(4):
            torque = self.kp * (target - self.data.qpos[self.qids]) - self.kd * self.data.qvel[self.vids]
            self.data.ctrl[self.aids] = np.clip(torque, -self.effort, self.effort)
            self.mj.mj_step(self.model, self.data)
        self.mj.mj_forward(self.model, self.data)
        self.steps += 1
        vel = np.zeros(6)
        self.mj.mj_objectVelocity(self.model, self.data, self.mj.mjtObj.mjOBJ_BODY, self.base, vel, 0)
        values = kinematics(self.data.xpos[self.base], self.data.xquat[self.base], vel[3:], vel[:3])
        reasons = []
        if self.steps > 1:
            if values[2] < self.cfg["terminationHeight"]:
                reasons.append("base_height")
            if values[-1] > self.cfg["terminationTilt"]:
                reasons.append("base_tilt")
            for i, c in enumerate(self.data.contact):
                if self.floor not in (c.geom1, c.geom2):
                    continue
                geom = c.geom2 if c.geom1 == self.floor else c.geom1
                body = self.model.geom_bodyid[geom]
                force = np.zeros(6)
                self.mj.mj_contactForce(self.model, self.data, i, force)
                if body not in self.allowed and self.data.xpos[body, 2] < self.cfg["terminationContactHeight"] \
                        and np.linalg.norm(force[:3]) > self.cfg["terminationContactForce"]:
                    reasons.append("illegal_ground_contact")
                    break
        if self.steps >= self.cfg["episodeLength"] - 1:
            reasons.append("time_out")
        return self.obs(), values, reasons, self.data.qpos[self.qids].copy()

    def render(self):
        self.camera.lookat[:] = self.data.xpos[self.base]
        self.camera.lookat[2] = max(.4, self.camera.lookat[2])
        self.camera.distance, self.camera.elevation = 3., -15
        views = []
        for azimuth in (135, 90):
            self.camera.azimuth = azimuth
            self.renderer.update_scene(self.data, camera=self.camera, scene_option=self.option)
            views.append(self.renderer.render().copy())
        return views

    def close(self):
        self.renderer.close()


def velocity_summary(rows):
    """Summarize a continuous rollout; all speed errors use the applied command."""
    a = np.asarray(rows, dtype=np.float64)
    stable, elapsed = [], 0.0
    dt = a[1, 0] - a[0, 0] if len(a) > 1 else .02
    for i, row in enumerate(a):
        changing = abs(row[1] - row[2]) > 1e-5 or (i > 0 and row[1] != a[i-1, 1])
        elapsed = 0.0 if changing else elapsed + dt
        stable.append(elapsed >= 1.0 and row[2] > .05)
    stable = np.asarray(stable)
    stop_times = []
    for i in range(1, len(a)):
        if a[i-1, 1] > 0 and a[i, 1] == 0:
            hold = 0.0
            for row in a[i:]:
                if row[1] > 0:
                    break
                quiet = np.hypot(row[3], row[4]) < .05 and row[8] < .1 and row[9] < .3
                hold = hold + dt if quiet and row[2] <= .05 else 0.0
                if hold >= .5 - 1e-6:
                    stop_times.append(float(row[0] - a[i, 0] + dt))
                    break
            else:
                hold = 0.0
            if hold < .5 - 1e-6:
                stop_times.append(None)
    tail = a[a[:, 0] > a[-1, 0] - 5]
    stopped_tail = len(tail) >= int(round(5 / dt)) and np.all(tail[:, 1:3] <= .05)
    drift = float(np.linalg.norm(np.diff(tail[:, 11:13], axis=0), axis=1).sum()) if stopped_tail else None
    error = float(a[stable, 6].mean()) if stable.any() else None
    quiet = bool(stopped_tail and np.hypot(tail[:, 3], tail[:, 4]).max() < .05
                 and tail[:, 8].mean() < .1 and tail[:, 9].mean() < .3)
    return dict(steady_vx_mae_m_s=error, stop_times_s=stop_times,
                final_5s_path_m=drift, final_stand_quiet=quiet,
                numeric_targets_met=bool(error is not None and error <= .12 and quiet
                    and drift <= .10 and stop_times
                    and all(value is not None and value <= 3.0 for value in stop_times)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("gym", "mujoco"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=int)
    parser.add_argument("--motion-file", type=Path, help="Use reference initial states; velocity task defaults to standing")
    parser.add_argument("--policy-npz", type=Path, help="Use exported NumPy actor for local MuJoCo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--command-profile", choices=("walk", "walk-stop", "restart", "joystick"))
    parser.add_argument("--duration", type=float, help="Velocity replay seconds; default 45")
    parser.add_argument("--no-video", action="store_true", help="Write CSV without camera/video")
    parser.add_argument("--viewer", action="store_true", help="Show the Isaac Gym viewer")
    parser.add_argument("--real-time", action="store_true")
    parser.add_argument("--joystick", type=int, default=0)
    parser.add_argument("--axis", type=int, default=1)
    parser.add_argument("--axis-sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--stop-button", type=int, default=0, help="Hold to command zero; -1 disables")
    args = parser.parse_args()
    if args.cases is not None and args.cases < 1:
        parser.error("cases must be positive")
    if args.duration is not None and (not np.isfinite(args.duration) or args.duration < .02):
        parser.error("duration must be finite and at least one control step")
    if args.engine == "gym":
        if args.policy_npz:
            parser.error("policy-npz is supported with mujoco only")
        import isaacgym  # Must precede torch.
    elif args.viewer:
        parser.error("viewer is supported with gym only")
    if not args.policy_npz:
        import torch
        torch.set_num_threads(4)
        torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    actor = (load_actor_numpy(args.policy_npz) if args.policy_npz else
             load_actor(args.checkpoint, "cuda:0" if args.engine == "gym" else "cpu"))
    velocity = actor.policy_config is not None
    if velocity:
        spec = actor.policy_config
        if (spec.get("task") != "HumanoidF1Velocity" or spec.get("version") not in (1, 2)
                or actor.observation_size != 96 or actor.algorithm != "ASE"):
            parser.error("Unsupported velocity policy schema")
        if args.engine != "gym":
            parser.error("Velocity policies currently replay in Isaac Gym only")
    elif actor.observation_size != 93 or args.command_profile or args.duration is not None:
        parser.error("Velocity commands require a velocity-conditioned checkpoint")
    if args.cases is None:
        args.cases = 3 if velocity else (5 if args.motion_file else 12)
    if velocity:
        args.command_profile = args.command_profile or "walk-stop"
        args.duration = args.duration or 45.0
    cfg_name = "humanoid_f1_velocity.yaml" if velocity else "humanoid_f1.yaml"
    cfg = load_env_config(ROOT / "ase/data/cfg" / cfg_name)
    cfg["env"].update(actor.control_config)
    if velocity:
        cfg["env"]["velocityTask"] = spec["velocity_settings"]
        control_dt = cfg["sim"]["dt"] * cfg["env"]["controlFrequencyInv"]
        if not np.isclose(control_dt, spec["control_dt"]):
            parser.error("Checkpoint control period differs from the replay configuration")
        cfg["env"]["episodeLength"] = max(2300, int(round(args.duration / control_dt)) + 2)
    print("REPLAY_CONTROL " + json.dumps(actor.control_config), flush=True)
    pad = None
    if velocity and args.command_profile == "joystick":
        pad = JoystickSpeed(args.joystick, args.axis, args.axis_sign, args.stop_button,
                            spec["velocity_settings"]["maxSpeed"])
        args.real_time = True
    args.output.mkdir(parents=True, exist_ok=True)
    names = joint_names()
    engine, video = None, None
    stem = "isaac" if args.engine == "gym" else "mujoco"
    summaries = []
    try:
        engine = (GymReplay(cfg, names, velocity, not args.no_video, args.viewer)
                  if args.engine == "gym" else MujocoReplay(cfg, names))
        if not args.no_video:
            video = Video(args.output / (stem + "_play.mp4"))
        velocity_columns = ["target_vx", "command_vx", "heading_vx", "heading_vy", "heading_yaw_rate",
                            "vx_error", "encoder_mask", "foot_speed", "leg_joint_speed_rms", "task_reward"]
        posture_columns = list(engine.task.velocity_posture_telemetry()) if velocity else []
        with (args.output / (stem + "_diag.csv")).open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["engine", "case", "init", "latent_mode", "step", "time_s", "base_x", "base_y",
                "base_z", "local_vx", "local_vy", "yaw_rate", "tilt_rad", "done", "termination",
                "action_rms", *(velocity_columns if velocity else []), *posture_columns, *["q_" + n for n in names]])
            for case in range(args.cases):
                state = initial_case(case, cfg, names, args.motion_file) if not velocity or args.motion_file else initial_case(0, cfg, names)
                obs = engine.reset(state)
                mode = ("switching" if case >= 8 and not velocity else "fixed") if actor.algorithm == "ASE" else "none"
                latent = Latents(args.seed + case, mode == "switching")
                origin = state["root"][:2].copy()
                rows = []
                steps = int(round(args.duration / engine.dt)) if velocity else cfg["env"]["episodeLength"] - 1
                wall_start = time.perf_counter()
                for step in range(steps):
                    if velocity:
                        speed = pad.read() if pad is not None else scripted_speed(step * control_dt, args.command_profile)
                        obs = engine.command_observation(speed)
                    z = latent.next()[None, :]
                    if not args.policy_npz:
                        z = torch.as_tensor(z, device=engine.device)
                    action = actor(obs, z)
                    action_rms = (float(np.sqrt(np.mean(action ** 2))) if args.policy_npz else
                                  float(action.square().mean().sqrt()))
                    obs, values, reasons, q = engine.step(action)
                    if velocity and step + 1 == steps and not reasons:
                        reasons = ["sequence_complete"]
                    done = bool(reasons)
                    telemetry = engine.task.velocity_telemetry()[0].tolist() if velocity else []
                    posture = engine.task.velocity_posture_telemetry() if velocity else {}
                    seconds = (step + 1) * engine.dt
                    writer.writerow([stem, case, state["init"], mode, step + 1, seconds,
                                     *values, int(done), "+".join(reasons), action_rms, *telemetry,
                                     *[float(posture[name][0]) for name in posture_columns], *q])
                    if velocity:
                        rows.append([seconds, *telemetry, *values[:2]])
                    if video is not None and (step % 2 == 0 or done):
                        label = ("F1 " + actor.algorithm + " epoch" + str(actor.epoch) + " | " + stem
                                 + " | case " + str(case+1) + "/" + str(args.cases) + " | z seed " + str(args.seed+case)
                                 + "\n" + state["init"] + " | {:.2f}s | ".format(seconds)
                                 + ("+".join(reasons) if done else "policy rollout"))
                        if velocity:
                            label += " | u {:.2f} c {:.2f} vx {:.2f}".format(*telemetry[:3])
                            if posture:
                                label += " | feet {:.0f}/{:.0f} deg".format(
                                    float(posture["left_foot_yaw_deg"][0]), float(posture["right_foot_yaw_deg"][0]))
                        video.write(engine.render(), label)
                    if args.real_time:
                        delay = wall_start + seconds - time.perf_counter()
                        if delay > 0:
                            time.sleep(delay)
                    if done:
                        break
                summary = dict(case=case, init=state["init"], latent=mode, latent_seed=args.seed+case,
                               seconds=round((step+1)*engine.dt, 3),
                               distance_m=round(float(np.linalg.norm(np.array(values[:2])-origin)), 3),
                               termination=reasons)
                if velocity:
                    summary.update(velocity_summary(rows))
                    summary["sequence_complete"] = reasons == ["sequence_complete"]
                    summary["numeric_targets_met"] &= summary["sequence_complete"]
                summaries.append(summary)
                print(json.dumps(summary), flush=True)
    finally:
        try:
            if video is not None:
                video.close()
        finally:
            try:
                if engine is not None:
                    engine.close()
            finally:
                if pad is not None:
                    pad.close()
    print(json.dumps({"engine": stem, "cases": len(summaries), "mean_duration_s":
        float(np.mean([s["seconds"] for s in summaries])), "timeouts":
        sum("time_out" in s["termination"] for s in summaries), "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
