"""Shared-avatar optimization through an explicit native-renderer adapter.

This module does not reconstruct a Gaussian avatar or replace native pipelines.
An adapter supplies render(view), gaussian_count, save(path), and parameters().
Rendered images must be RGB CHW tensors in [0,1], on a white background.
"""

import torch
from torch.nn import functional as F


def ssim_loss(image, target):
    """Standard 11x11 Gaussian-window SSIM loss, averaged over RGB/pixels."""
    coords = torch.arange(11, device=image.device, dtype=image.dtype) - 5
    weights = torch.exp(-coords.square() / (2 * 1.5**2))
    weights /= weights.sum()
    window = (weights[:, None] * weights[None, :])[None, None].expand(3, 1, 11, 11)
    image, target = image[None], target[None]
    blur = lambda x: F.conv2d(x, window, padding=5, groups=3)
    mu_x, mu_y = blur(image), blur(target)
    var_x, var_y = (
        blur(image * image) - mu_x.square(),
        blur(target * target) - mu_y.square(),
    )
    covariance = blur(image * target) - mu_x * mu_y
    score = ((2 * mu_x * mu_y + 0.01**2) * (2 * covariance + 0.03**2)) / (
        (mu_x.square() + mu_y.square() + 0.01**2) * (var_x + var_y + 0.03**2)
    )
    return 1 - score.mean()


def optimize_avatar(avatar, views, targets, optimizer, steps=500, seed=2026):
    """Optimize existing parameters only, with 0.6 L1 + 0.4 SSIM loss."""
    if not views or len(views) != len(targets) or steps < 1:
        raise ValueError(
            "Provide matching nonempty view/target lists and a positive step count."
        )
    count = avatar.gaussian_count
    parameter_ids = {id(p) for p in avatar.parameters()}
    if hasattr(avatar, "begin_optimization"):
        avatar.begin_optimization()
    rng = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(steps):
        i = int(torch.randint(len(views), (1,), generator=rng))
        prediction = avatar.render(views[i])
        target = targets[i].detach().to(prediction)
        if (
            prediction.shape != target.shape
            or prediction.ndim != 3
            or prediction.shape[0] != 3
        ):
            raise ValueError(
                "Native render and pseudo-target must be matching RGB CHW tensors."
            )
        loss = 0.6 * F.l1_loss(prediction, target) + 0.4 * ssim_loss(prediction, target)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite avatar loss.")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if hasattr(avatar, "after_step"):
            avatar.after_step()
        if (
            avatar.gaussian_count != count
            or {id(p) for p in avatar.parameters()} != parameter_ids
        ):
            raise RuntimeError(
                "Adapter changed Gaussian count or replaced parameters during refinement."
            )
        losses.append(float(loss.detach()))
    return losses


def generate_targets(refiner, avatar, views, reference, seed=2026):
    """Refine frozen native renders; views must use a 512x512 white canvas."""
    if not views:
        raise ValueError("No views supplied.")
    dtype = next(refiner.parameters()).dtype
    device = next(refiner.parameters()).device
    rng = torch.Generator(device=device).manual_seed(seed)
    targets = []
    with torch.inference_mode():
        features = refiner.encode_reference(reference.to(device=device, dtype=dtype))
        for view in views:
            image = avatar.render(view)
            if image.shape != (3, 512, 512):
                raise ValueError(
                    "Adapter must render a 512x512 white canvas; camera intrinsics must match it."
                )
            refined = refiner(
                image[None].to(device=device, dtype=dtype) * 2 - 1,
                reference_features=features,
                generator=rng,
            )
            targets.append((refined[0].float() * 0.5 + 0.5).cpu())
    # Convert inference tensors into ordinary detached tensors for autograd losses.
    return [target.clone() for target in targets]
