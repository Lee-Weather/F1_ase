"""Replay an F1 ASE checkpoint in Isaac Gym or MuJoCo without training.

Example: python ase/replay_f1.py --engine mujoco --checkpoint MODEL --output DIR
Both engines use the same deterministic actor, latent seeds and initial states.
Outputs are one two-view video and one CSV per engine; no exported model needed.
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
import xml.etree.ElementTree as ET

import numpy as np
import yaml

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

    def linear(x, name):
        return F.linear(x, weights[name + ".weight"], weights[name + ".bias"])

    def actor(obs, latent):
        with torch.no_grad():
            obs = torch.clamp((obs - mean) / scale, -5, 5)
            style = F.relu(linear(latent, "actor_mlp._style_mlp.0"))
            style = F.relu(linear(style, "actor_mlp._style_mlp.2"))
            style = torch.tanh(linear(style, "actor_mlp._style_dense"))
            hidden = torch.cat((obs, style), dim=-1)
            for i in range(3):
                hidden = F.relu(linear(hidden, "actor_mlp._dense_layers." + str(i)))
            return linear(hidden, "mu").clamp(-1, 1)

    print(json.dumps({"checkpoint_epoch": ckpt["epoch"], "frame": ckpt.get("frame"),
                      "actor_observations": int(mean.numel()), "deterministic": True}), flush=True)
    actor.epoch = int(ckpt["epoch"])
    return actor


def joint_names():
    return [j.get("name") for j in ET.parse(ASSET).iter("joint")]


def initial_case(index, cfg, names):
    default = np.array([cfg["env"]["defaultJointPos"][n] for n in names])
    root = np.array([0., 0., cfg["env"]["initialHeight"], 1., 0., 0., 0.])  # wxyz
    state = dict(q=default, qd=np.zeros(29), root=root, velocity=np.zeros(6), init="default")
    if 4 <= index < 8:
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
    def __init__(self, cfg, names):
        from isaacgym import gymapi
        import torch
        from env.tasks.humanoid_f1 import HumanoidF1
        params = gymapi.SimParams()
        params.dt, params.substeps = cfg["sim"]["dt"], cfg["sim"]["substeps"]
        params.use_gpu_pipeline = params.physx.use_gpu = True
        for key, value in cfg["sim"]["physx"].items():
            if hasattr(params.physx, key):
                setattr(params.physx, key, value)
        cfg["env"].update(numEnvs=1, stateInit="Default", enableCameraSensors=True)
        cfg["env"]["motion_file"] = str(ROOT / "ase/data/motions/f1/dataset_f1.yaml")
        cfg["env"]["asset"]["assetRoot"] = str(ROOT / "ase/data/assets")
        self.task = HumanoidF1(cfg, params, gymapi.SIM_PHYSX, "cuda", 0, True)
        assert list(self.task._dof_names) == names, "Actor joint order differs from asset"
        self.gym, self.sim, self.env = self.task.gym, self.task.sim, self.task.envs[0]
        properties = gymapi.CameraProperties()
        properties.width, properties.height = 640, 480
        self.cameras = [self.gym.create_camera_sensor(self.env, properties) for _ in range(2)]
        if min(self.cameras) < 0:
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
        self.gym.destroy_sim(self.sim)


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
        limits = {j.get("name"): float(j.find("limit").get("effort"))
                  for j in urdf.findall("joint") if j.get("type") == "revolute"}
        self.effort = np.array([limits[n] for n in names])
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
        import torch
        rot = self.data.xmat[self.base].reshape(3, 3)
        velocity = np.zeros(6)
        self.mj.mj_objectVelocity(self.model, self.data, self.mj.mjtObj.mjOBJ_BODY, self.base, velocity, 0)
        obs = np.r_[rot.T @ velocity[:3], rot.T @ [0., 0., -1.],
                    self.data.qpos[self.qids] - self.default, self.data.qvel[self.vids], self.action]
        return torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)

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
        self.action = action[0].cpu().numpy()
        target = self.default + self.cfg["actionScale"] * self.action
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("gym", "mujoco"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.cases < 1:
        parser.error("cases must be positive")
    if args.engine == "gym":
        import isaacgym  # Must precede torch.
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / "ase/data/cfg/humanoid_f1.yaml").read_text())
    names = joint_names()
    engine = GymReplay(cfg, names) if args.engine == "gym" else MujocoReplay(cfg, names)
    actor = load_actor(args.checkpoint, engine.device)
    stem = "isaac" if args.engine == "gym" else "mujoco"
    video = Video(args.output / (stem + "_play.mp4"))
    summaries = []
    try:
        with (args.output / (stem + "_diag.csv")).open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["engine", "case", "init", "latent_mode", "step", "time_s", "base_x", "base_y",
                "base_z", "local_vx", "local_vy", "yaw_rate", "tilt_rad", "done", "termination",
                "action_rms", *["q_" + n for n in names]])
            for case in range(args.cases):
                state = initial_case(case, cfg, names)
                obs = engine.reset(state)
                mode = "switching" if case >= 8 else "fixed"
                latent = Latents(args.seed + case, mode == "switching")
                origin = state["root"][:2].copy()
                for step in range(cfg["env"]["episodeLength"] - 1):
                    z = torch.as_tensor(latent.next(), device=engine.device).unsqueeze(0)
                    action = actor(obs, z)
                    obs, values, reasons, q = engine.step(action)
                    done = bool(reasons)
                    writer.writerow([stem, case, state["init"], mode, step + 1, (step+1)*engine.dt,
                                     *values, int(done), "+".join(reasons),
                                     float(action.square().mean().sqrt()), *q])
                    if step % 2 == 0 or done:
                        label = ("F1 ASE epoch" + str(actor.epoch) + " | " + stem + " | case " + str(case+1) + "/" + str(args.cases)
                                 + " | " + mode + " latent\n" + state["init"] + " | {:.2f}s | ".format((step+1)*engine.dt)
                                 + ("+".join(reasons) if done else "policy rollout"))
                        video.write(engine.render(), label)
                    if done:
                        break
                summary = dict(case=case, init=state["init"], latent=mode, seconds=round((step+1)*engine.dt, 3),
                               distance_m=round(float(np.linalg.norm(np.array(values[:2])-origin)), 3),
                               termination=reasons)
                summaries.append(summary)
                print(json.dumps(summary), flush=True)
    finally:
        try:
            video.close()
        finally:
            engine.close()
    print(json.dumps({"engine": stem, "cases": len(summaries), "mean_duration_s":
        float(np.mean([s["seconds"] for s in summaries])), "timeouts":
        sum("time_out" in s["termination"] for s in summaries), "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
