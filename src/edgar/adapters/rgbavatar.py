"""RGBAvatar refinement using its native deformation, renderer and color update.

Camera/optimizer conventions adapted from the supplied RGBAvatar application.
RGBAvatar copyright (c) 2025 Linzhou Li; see licenses/RGBAvatar-MIT.txt.
"""

import math
import random
from pathlib import Path

import torch

from .common import (
    canvas,
    native_root,
    namespace_package,
    optimizer_parameters,
    refinement_rates,
)


def orbit(view, angle):
    w2c = view["cam_view_mat"][0]
    c2w = torch.linalg.inv(w2c)
    projection = view["cam_proj_mat"][0] @ c2w
    theta = math.radians(angle)
    c, s = math.cos(theta), math.sin(theta)
    rotation = w2c.new_tensor([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])
    new_c2w = rotation @ c2w
    new_w2c = torch.linalg.inv(new_c2w)
    return {
        "cam_view_mat": new_w2c[None],
        "cam_proj_mat": (projection @ new_w2c)[None],
        "cam_position": new_c2w[:3, 3][None].contiguous(),
    }


class RGBAvatarAdapter:
    def __init__(self, model, dataset, optimizer, recon_config, render):
        self.model, self.dataset, self.optimizer = model, dataset, optimizer
        self.config, self.native_render = recon_config, render
        self.optimizing = False
        self.last_render = None

    @property
    def gaussian_count(self):
        return int(self.model.num_gaussian)

    def parameters(self):
        return optimizer_parameters(self.optimizer)

    def render(self, view):
        index, angle = view
        row = self.dataset[index]
        batch = {
            key: row[key][None].cuda()
            for key in (
                "mesh",
                "blend_weight",
                "cam_view_mat",
                "cam_proj_mat",
                "cam_position",
            )
        }
        camera = orbit(batch, angle)
        gaussians = self.model.gaussian_deform_batch(
            batch["mesh"], batch["blend_weight"]
        )
        self.last_render = self.native_render(
            self.dataset.fov_x,
            self.dataset.fov_y,
            self.dataset.image_width,
            self.dataset.image_height,
            camera["cam_view_mat"],
            camera["cam_proj_mat"],
            camera["cam_position"],
            torch.ones((1, 3), device="cuda"),
            gaussians,
        )
        image = self.last_render["color"]
        if self.optimizing:
            image = image.clamp(0, 1)
        return canvas(image)

    def begin_optimization(self):
        self.optimizing = True

    def after_step(self):
        if self.config.use_fast_forward:
            self.model.fast_forward(
                self.last_render["est_color"], self.last_render["est_weight"]
            )
        self.last_render = None

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        self.model.save_ply(str(path / "model.ply"))


def create(config):
    root = native_root(config)
    namespace_package("model", root / "model")
    namespace_package("dataset", root / "dataset")
    from model.binding import FLAMEBindingModel
    from dataset.nersemble_ga_dataset import NeRSembleGADataset
    from submodules.flame import FLAME, FlameConfig
    from utils import Struct
    from nvdiffrast import torch as dr
    import yaml
    from .rgbavatar_renderer import render_gs_batch

    settings = yaml.safe_load(Path(config["native_config"]).read_text())
    flame_config = FlameConfig()
    flame_config.dtype = torch.float32
    flame_config.flame_geom_path = config["flame_model"]
    flame = FLAME(flame_config).cuda()
    dataset = NeRSembleGADataset(flame, config["transforms"], train=False)
    context = (
        dr.RasterizeGLContext()
        if config.get("raster_context", "cuda") == "gl"
        else dr.RasterizeCudaContext()
    )
    model = FLAMEBindingModel(Struct(**settings["model"]), flame, context)
    model.initialize()
    model.load_ply(config["baseline"], train=True)
    recon = Struct(**settings["train"]["recon"])
    groups = model.training_params(recon)
    optimizer = torch.optim.Adam(
        params=sum((list(g) for g in groups), []), lr=0, eps=1e-15
    )
    refinement_rates(optimizer, recon)
    cameras = config.get("camera_indices", [])
    if cameras:
        import json

        frames = json.loads(Path(config["transforms"]).read_text())["frames"]
        eligible = [
            i for i, frame in enumerate(frames) if frame["camera_index"] in cameras
        ]
    else:
        eligible = list(range(len(dataset)))
    if not eligible:
        raise ValueError("No training frames match camera_indices.")
    count = int(config.get("camera_count", 50))
    angles = config.get("angles", [-60, -30, 30, 60])
    if count < 1 or not angles:
        raise ValueError("Provide a positive camera_count and nonempty angles.")
    rng = random.Random(config.get("seed", 2026))
    sampled = [rng.choice(eligible) for _ in range(count)]
    views = [(index, float(angle)) for index in sampled for angle in angles]
    return (
        RGBAvatarAdapter(model, dataset, optimizer, recon, render_gs_batch),
        views,
        optimizer,
    )
