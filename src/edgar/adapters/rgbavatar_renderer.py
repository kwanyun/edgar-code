"""RGBAvatar native batch rendering, with unrelated training imports removed.
Copyright (c) 2025 Linzhou Li. See licenses/RGBAvatar-MIT.txt.
"""

from __future__ import annotations
import math
import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)


def render_gs_batch(
    fov_x: float,
    fov_y: float,
    width: int,
    height: int,
    cam_view_mats: torch.Tensor,
    cam_proj_mats: torch.Tensor,
    cam_positions: torch.Tensor,
    bg_colors: torch.Tensor,
    gs,
    target_image: torch.Tensor = None,
    sh_degree: int = 0,
    scaling_modifier: float = 1.0,
) -> dict[str, torch.Tensor]:
    screenspace_points = (
        torch.zeros_like(
            gs.xyz, dtype=gs.xyz.dtype, requires_grad=True, device=gs.xyz.device
        )
        + 0
    )
    if screenspace_points.requires_grad:
        screenspace_points.retain_grad()
    tanfovx = math.tan(fov_x * 0.5)
    tanfovy = math.tan(fov_y * 0.5)
    batch_size = gs.xyz.shape[0]
    cam_view_mats = cam_view_mats.reshape(batch_size, 4, 4).transpose(1, 2)
    cam_proj_mats = cam_proj_mats.reshape(batch_size, 4, 4).transpose(1, 2)
    cam_positions = cam_positions.reshape(batch_size, 3)
    bg_colors = bg_colors.reshape(batch_size, 3)
    bs = gs.xyz.shape[0]
    color_list = []
    alpha_list = []
    est_color_list = []
    est_weight_list = []
    radii_list = []
    for i in range(bs):
        cam_view_mat = cam_view_mats[i].contiguous()
        cam_proj_mat = cam_proj_mats[i].contiguous()
        cam_position = cam_positions[i].contiguous()
        bg_color = bg_colors[i].contiguous()
        raster_settings = GaussianRasterizationSettings(
            image_height=height,
            image_width=width,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=cam_view_mat,
            projmatrix=cam_proj_mat,
            sh_degree=sh_degree,
            campos=cam_position,
            prefiltered=False,
            debug=False,
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        (color, alpha, est_color, est_weight, radii) = rasterizer(
            means3D=gs.xyz[i],
            means2D=screenspace_points[i],
            shs=gs.sh[i],
            colors_precomp=None,
            opacities=gs.opacity[i],
            scales=gs.scaling[i],
            rotations=gs.rotation[i],
            cov3D_precomp=None,
            target_image=target_image[i] if target_image is not None else None,
        )
        color_list.append(color)
        alpha_list.append(alpha)
        est_color_list.append(est_color)
        est_weight_list.append(est_weight)
        radii_list.append(radii)
    return {
        "color": torch.stack(color_list),
        "alpha": torch.stack(alpha_list),
        "est_color": torch.stack(est_color_list),
        "est_weight": torch.stack(est_weight_list),
        "radii": torch.stack(radii_list),
    }
