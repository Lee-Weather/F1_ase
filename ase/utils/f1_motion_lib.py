"""F1 tracking NPZ motions, in simulator name order and xyzw quaternions."""
from pathlib import Path
import hashlib

import numpy as np
import torch
import yaml


class F1MotionLib:
    def __init__(self, motion_file, dof_names, body_names, key_body_ids, device):
        self._device = device
        path = Path(motion_file)
        entries = yaml.safe_load(path.read_text())["motions"] if path.suffix == ".yaml" else [{"file": path.name, "weight": 1.0}]
        clips, lengths, frames, weights = [], [], [], []
        self._motion_files = []
        self._manifest = []
        for entry in entries:
            file = path.parent / entry["file"]
            with np.load(str(file), allow_pickle=False) as raw:
                joints = raw["joint_names"].tolist()
                bodies = raw["body_names"].tolist()
                ji = [joints.index(name) for name in dof_names]
                bi = [bodies.index(name) for name in body_names]
                n = len(raw["joint_pos"])
                fps = float(raw["fps"].reshape(-1)[0])
                if n < 2 or fps <= 0:
                    raise ValueError("Invalid motion: " + str(file))
                arrays = {
                    "joint_pos": raw["joint_pos"][:, ji],
                    "joint_vel": raw["joint_vel"][:, ji],
                    "body_pos": raw["body_pos_w"][:, bi],
                    "body_rot": raw["body_quat_w"][:, bi][..., [1, 2, 3, 0]],
                    "body_vel": raw["body_lin_vel_w"][:, bi],
                    "body_ang_vel": raw["body_ang_vel_w"][:, bi],
                }
                if not all(np.isfinite(a).all() for a in arrays.values()):
                    raise ValueError("Non-finite motion: " + str(file))
                clips.append({k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in arrays.items()})
                frames.append(n)
                lengths.append((n - 1) / fps)
                weights.append(float(entry["weight"]))
                self._motion_files.append(str(file))
                self._manifest.append({"file": file.name, "weight": float(entry["weight"]),
                                       "frames": n, "fps": fps,
                                       "sha256": hashlib.sha256(file.read_bytes()).hexdigest()})
        self._motion_lengths = torch.tensor(lengths, device=device)
        self._motion_num_frames = torch.tensor(frames, dtype=torch.long, device=device)
        self._starts = torch.cumsum(self._motion_num_frames, 0) - self._motion_num_frames
        self._data = {k: torch.cat([c[k] for c in clips]) for k in clips[0]}
        self._weights = torch.tensor(weights, dtype=torch.float32, device=device)
        if (self._weights < 0).any() or self._weights.sum() <= 0:
            raise ValueError("Motion weights must be nonnegative with a positive sum")
        self._weights /= self._weights.sum()
        self._key_body_ids = torch.as_tensor(key_body_ids, dtype=torch.long, device=device)
        print("F1 motions: {} clips, {:.2f}s".format(len(clips), sum(lengths)))

    def num_motions(self):
        return len(self._motion_files)

    def get_manifest(self):
        return [dict(entry) for entry in self._manifest]

    def get_total_length(self):
        return self._motion_lengths.sum()

    def get_motion_length(self, motion_ids):
        return self._motion_lengths[motion_ids]

    def sample_motions(self, n):
        return torch.multinomial(self._weights, n, replacement=True)

    def sample_time(self, motion_ids, truncate_time=0.0):
        truncate_time = 0.0 if truncate_time is None else truncate_time
        lengths = self._motion_lengths[motion_ids]
        return torch.rand_like(lengths) * torch.clamp(lengths - truncate_time, min=0.0)

    def get_motion_state(self, motion_ids, motion_times):
        state = self.get_body_state(motion_ids, motion_times)
        return (state["body_pos"][:, 0], state["body_rot"][:, 0], state["joint_pos"],
                state["body_vel"][:, 0], state["body_ang_vel"][:, 0], state["joint_vel"],
                state["body_pos"][:, self._key_body_ids])

    def get_body_state(self, motion_ids, motion_times):
        # Clip at the ends; short clips are never silently treated as seamless loops.
        phase = torch.clamp(motion_times / self._motion_lengths[motion_ids], 0.0, 1.0)
        frame = phase * (self._motion_num_frames[motion_ids] - 1)
        i0 = frame.long()
        i1 = torch.minimum(i0 + 1, self._motion_num_frames[motion_ids] - 1)
        blend = frame - i0
        i0, i1 = i0 + self._starts[motion_ids], i1 + self._starts[motion_ids]
        state = {}
        for key, values in self._data.items():
            a, b = values[i0], values[i1]
            t = blend.reshape((-1,) + (1,) * (a.ndim - 1))
            if key == "body_rot":
                dot = (a * b).sum(-1, keepdim=True)
                b = torch.where(dot < 0, -b, b)
                angle = torch.acos(torch.clamp(dot.abs(), max=1.0))
                denom = torch.sin(angle).clamp_min(1e-7)
                spherical = (torch.sin((1 - t) * angle) * a + torch.sin(t * angle) * b) / denom
                linear = (1 - t) * a + t * b
                q = torch.where(angle < 1e-4, linear, spherical)
                state[key] = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            else:
                state[key] = (1 - t) * a + t * b
        return state
