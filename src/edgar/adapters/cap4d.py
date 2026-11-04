"""Refine an existing CAP4D avatar with the native CAP4D renderer and optimizers."""

import copy
import hashlib
import math
import random
import os
import shutil
from pathlib import Path

import torch
from torch.nn import functional as F

from .common import canvas, native_root, optimizer_parameters, refinement_rates


def training_state_digest(model):
    """Identify the tracked sequence to which per-frame neck offsets belong."""
    digest = hashlib.sha256()
    for name in ("expr", "eye_rot", "rot", "tra"):
        value = model.flame_param[name].detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def restore_training_offsets(model, state):
    """Restore EDGAR's optional per-frame state for the same tracked sequence.

    For reenactment with a different driving sequence, use native restore alone.
    """
    extra = state.get("edgar_training_state")
    if extra is None:
        return
    if model.static_neck or extra["sequence_sha256"] != training_state_digest(model):
        raise ValueError("Saved neck offsets require the original tracked sequence.")
    value = extra["neck_rot_offset"].to(model.neck_rot_offset.weight)
    if value.shape != model.neck_rot_offset.weight.shape:
        raise ValueError("Saved neck offsets do not match the tracked frame count.")
    with torch.no_grad():
        model.neck_rot_offset.weight.copy_(value)


def orbit(camera, model, angle):
    device = camera.rt.device
    position = camera.get_center().detach().to(device)
    center = model.face_center.mean(0).detach().to(device)
    theta = math.radians(angle)
    c, s = math.cos(theta), math.sin(theta)
    rotation = torch.tensor(
        [[c, 0, s], [0, 1, 0], [-s, 0, c]], device=device, dtype=torch.float32
    )
    new_position = rotation @ (position - center) + center
    z = F.normalize(center - new_position, dim=0)
    x = F.normalize(
        torch.cross(torch.tensor([0.0, 1.0, 0.0], device=device), z, dim=0), dim=0
    )
    y = torch.cross(z, x, dim=0)
    R = torch.stack((x, y, z))
    result = copy.deepcopy(camera)
    result.rt = torch.eye(4, device=device)
    result.rt[:3, :3] = R
    result.rt[:3, 3] = -R @ new_position
    return result


class CAP4DOptimizer:
    def __init__(self, model):
        self.model = model
        self.param_groups = list(model.optimizer.param_groups)
        if not model.static_neck:
            self.param_groups += list(model.neck_optimizer.param_groups)

    def zero_grad(self, set_to_none=True):
        self.model.optimizer.zero_grad(set_to_none=set_to_none)
        if not self.model.static_neck:
            self.model.neck_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.model.eval()
        self.model.optimizer_step()


class CAP4DAdapter:
    def __init__(self, model, cameras, optimizer, render, iteration):
        self.model, self.cameras, self.optimizer, self.native_render = (
            model,
            cameras,
            optimizer,
            render,
        )
        self.iteration, self.steps = int(iteration), 0
        self.optimizing = False

    @property
    def gaussian_count(self):
        return int(self.model._xyz.shape[0])

    def parameters(self):
        return optimizer_parameters(self.optimizer)

    def begin_optimization(self):
        self.optimizing = True

    def render(self, view):
        index, angle = view
        camera = self.cameras[index]
        if self.model.binding is not None:
            self.model.select_mesh_by_timestep(camera.timestep)
        # Preserve the original application's mode ordering around deformation.
        if self.optimizing:
            self.model.train()
        camera = orbit(camera, self.model, angle)
        return canvas(
            self.native_render(camera, self.model, torch.ones(3, device="cuda"))[
                "render"
            ]
        )

    def after_step(self):
        self.steps += 1

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        iteration = self.iteration + self.steps
        state = self.model.capture()
        # Native capture omits these optimized, sequence-specific parameters.
        # Its restore ignores extra keys, keeping reenactment compatibility.
        if not self.model.static_neck:
            state["edgar_training_state"] = {
                "sequence_sha256": training_state_digest(self.model),
                "neck_rot_offset": self.model.neck_rot_offset.weight.detach().cpu(),
            }
        torch.save((state, iteration), path / f"chkpnt{iteration}.pth")


def create(config):
    repository = native_root(config)
    from omegaconf import OmegaConf
    from gaussianavatars.scene.cap4d_gaussian_model import CAP4DGaussianModel
    from gaussianavatars.scene.scene import Scene
    from gaussianavatars.gaussian_renderer.gsplat_renderer import render

    workspace = Path(config["_workspace"])
    workspace.mkdir(parents=True, exist_ok=False)
    # Native CAP4D resolves its licensed assets relative to the working directory.
    # Keep a private compatible copy in this run; never overwrite the repository.
    assets = workspace / "data" / "assets"
    shutil.copytree(repository / "data" / "assets", assets)
    from .numpy_pickle import convert_geometry_copy

    convert_geometry_copy(assets / "flame" / "flame2023_no_jaw.pkl")
    os.chdir(workspace)
    settings = OmegaConf.load(config["native_config"])
    model = CAP4DGaussianModel(settings.model_params)
    scene = Scene(
        str(workspace), model, source_paths=config["source_paths"], shuffle=False
    )
    model.training_setup(settings.opt_params)
    from . import numpy_pickle

    state, iteration = torch.load(
        config["baseline"],
        map_location="cuda",
        pickle_module=numpy_pickle,
        weights_only=False,
    )
    model.restore(state, settings.opt_params)
    restore_training_offsets(model, state)
    refinement_rates(model.optimizer)
    optimizer = CAP4DOptimizer(model)
    cameras = scene.getTrainCameras()
    count = int(config.get("camera_count", 50))
    angles = config.get("angles", [-60, -30, 30, 60])
    if count < 1 or count > len(cameras) or not angles:
        raise ValueError(
            "camera_count must fit the training cameras; angles must be nonempty."
        )
    rng = random.Random(config.get("seed", 2026))
    sampled = rng.sample(list(range(len(cameras))), count)
    # Freeze original camera objects; mesh-specific orbit is recomputed each render.
    selected = [cameras[i] for i in sampled]
    views = [(i, float(angle)) for i in range(len(selected)) for angle in angles]
    return CAP4DAdapter(model, selected, optimizer, render, iteration), views, optimizer
