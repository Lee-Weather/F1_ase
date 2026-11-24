"""Measure full footfalls and sliding in the F1 flat-ground task."""

import math
import struct
import xml.etree.ElementTree as ET

import numpy as np
import torch


def load_foot_vertices(mjcf, body_names):
    """Read the F1 foot collision meshes in their respective link frames."""
    root = ET.parse(mjcf).getroot()
    compiler = root.find("compiler")
    mesh_dir = mjcf.parent / compiler.get("meshdir", ".")
    feet = []
    for name in body_names:
        geom = root.find(f".//geom[@name='{name}_collision_0']")
        mesh = root.find(f"asset/mesh[@name='{geom.attrib['mesh']}']")
        data = (mesh_dir / mesh.attrib["file"]).read_bytes()
        triangles, = struct.unpack_from("<I", data, 80)
        if len(data) != 84 + 50 * triangles:
            raise ValueError(f"Expected a binary STL for {name}")
        vertices = np.array([record[3:12] for record in struct.iter_unpack("<12fH", data[84:])])
        vertices = np.unique(vertices.reshape(-1, 3), axis=0)
        vertices *= np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
        origin = np.fromstring(geom.attrib["pos"], sep=" ")
        quat = np.fromstring(geom.attrib["quat"], sep=" ")
        w, *xyz = quat / np.linalg.norm(quat)
        vertices += 2 * np.cross(xyz, np.cross(xyz, vertices) + w * vertices)
        feet.append(vertices + origin)
    return np.stack(feet)


def world_foot_vertices(vertices, position, rotation):
    offsets = vertices.expand(position.shape[0], -1, -1, -1)
    xyz = rotation[:, :, None, :3].expand_as(offsets)
    w = rotation[:, :, None, 3:4]
    rotated = offsets + 2 * torch.cross(xyz, torch.cross(xyz, offsets, dim=-1) + w * offsets, dim=-1)
    return rotated + position[:, :, None, :]


class AlternatingGait:
    def __init__(self, count, device, dt, config, vertices):
        self.config = config
        self.dt = dt
        self.vertices = torch.as_tensor(vertices, dtype=torch.float32, device=device)
        self.min_air_steps = math.ceil(config["minSwingSeconds"] / dt - 1e-5)
        self.memory_steps = math.ceil(config["alternationMemorySeconds"] / dt - 1e-5)
        self.previous_live = torch.zeros(count, dtype=torch.bool, device=device)
        self.previous_support = torch.zeros((count, 2), dtype=torch.bool, device=device)
        self.previous_x = torch.zeros((count, 2), device=device)
        self.previous_points = self.vertices.new_zeros((count, *self.vertices.shape))
        self.air_steps = torch.zeros((count, 2), dtype=torch.long, device=device)
        self.opposite_support_steps = torch.zeros_like(self.air_steps)
        self.takeoff_x = torch.zeros_like(self.previous_x)
        self.peak_clearance = torch.zeros_like(self.previous_x)
        self.last_side = torch.full((count,), -1, dtype=torch.long, device=device)
        self.alternation_age = torch.full_like(self.last_side, self.memory_steps)

    def reset(self, ids):
        self.previous_live[ids] = False
        self.air_steps[ids] = 0
        self.opposite_support_steps[ids] = 0
        self.peak_clearance[ids] = 0
        self.last_side[ids] = -1
        self.alternation_age[ids] = self.memory_steps

    def step(self, position, rotation, force_z, live):
        cfg = self.config
        points = world_foot_vertices(self.vertices, position, rotation)
        clearance = points[..., 2].amin(dim=-1)
        supported = force_z >= cfg["minContactForceN"]
        continuing = (live & self.previous_live)[:, None]
        takeoff = continuing & self.previous_support & ~supported
        airborne = continuing & ~supported & (takeoff | (self.air_steps > 0))
        self.takeoff_x = torch.where(takeoff, self.previous_x, self.takeoff_x)
        landing = continuing & supported & (self.air_steps >= self.min_air_steps)
        complete = (landing & (self.peak_clearance >= cfg["minClearanceM"])
                    & (position[..., 0] - self.takeoff_x >= cfg["minAdvanceM"])
                    & (self.opposite_support_steps >= cfg["minOppositeSupportFraction"] * self.air_steps))
        self.air_steps = (self.air_steps + 1) * airborne
        self.opposite_support_steps = (self.opposite_support_steps + supported.flip(dims=(1,))) * airborne
        self.peak_clearance = torch.maximum(self.peak_clearance, clearance) * airborne

        single = complete.sum(dim=-1) == 1
        side = complete[:, 1].long()
        alternating = single & (self.last_side >= 0) & (side != self.last_side)
        self.last_side = torch.where(live, torch.where(single, side, self.last_side), -1)
        # A repeated footfall clears the speed multiplier. The next accepted
        # landing of the other foot can begin a new alternation.
        self.alternation_age = (self.alternation_age + 1).clamp(max=self.memory_steps)
        self.alternation_age = torch.where(~live | single, self.memory_steps, self.alternation_age)
        self.alternation_age = torch.where(alternating, 0, self.alternation_age)
        recent = 1 - self.alternation_age / self.memory_steps

        slip, stance = self._stance_slip(points, clearance, supported, continuing)
        self.previous_points.copy_(points)
        self.previous_x.copy_(position[..., 0])
        self.previous_support.copy_(supported)
        self.previous_live.copy_(live)
        return recent, alternating, complete, slip, stance

    def _stance_slip(self, points, clearance, supported, continuing):
        depth = self.config["stanceVertexHeightM"]
        # Compare the same material vertices while both samples are loaded and
        # the vertices are near the bottom of the foot at both sample times.
        previous_z = self.previous_points[..., 2]
        near = ((points[..., 2] <= clearance[..., None] + depth)
                & (previous_z <= previous_z.amin(dim=-1, keepdim=True) + depth))
        selected = near & (continuing & supported & self.previous_support)[..., None]
        speeds = (points[..., :2] - self.previous_points[..., :2]).norm(dim=-1) / self.dt
        count = selected.sum(dim=-1)
        return (speeds * selected).sum(dim=-1) / count.clamp(min=1), count > 0

    def moving_reward(self, speed_reward, recent, alternating, slip, stance):
        cfg = self.config
        slip_cost = (slip / cfg["slipSpeedScaleMps"]).square().clamp(max=1)
        slip_cost = slip_cost.sum(dim=-1) / stance.sum(dim=-1).clamp(min=1)
        floor = cfg["speedRewardFloor"]
        return (speed_reward * (floor + (1 - floor) * recent)
                + cfg["alternatingStepBonus"] * alternating.float() - cfg["stanceSlipWeight"] * slip_cost)
