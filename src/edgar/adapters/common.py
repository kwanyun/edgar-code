"""Native-pipeline loading and differentiable image preparation."""

import os
import sys
import types
from pathlib import Path

from torch.nn import functional as F


def native_root(config):
    root = Path(config["repository"]).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    sys.path.insert(0, str(root))
    # Native assets use paths relative to their own repository.
    os.chdir(root)
    return root


def namespace_package(name, path):
    """Load native submodules without importing unrelated training entry points."""
    if name in sys.modules:
        previous = list(getattr(sys.modules[name], "__path__", []))
        if previous != [str(path)]:
            raise RuntimeError(
                f"Module name {name!r} is already in use. Run the adapter in a fresh process."
            )
        return
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package


def canvas(image):
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Expected a native RGB CHW rendering.")
    _, h, w = image.shape
    dh, dw = max(h, w) - h, max(h, w) - w
    image = F.pad(image, (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2), value=1)
    return F.interpolate(
        image[None], size=(512, 512), mode="bilinear", align_corners=False
    )[0]


def optimizer_parameters(optimizer):
    return [p for group in optimizer.param_groups for p in group["params"]]


def refinement_rates(optimizer, config=None):
    rates = {
        "xyz": 1.6e-6,
        "opacity": 0.025,
        "scaling": 0.005,
        "rotation": 0.001,
        "f_dc": 0.0025,
        "f_rest": 0.0025 / 20,
    }
    if config is not None:
        rates.update(
            xyz_b=1.6e-6 * config.position_b_lr_scale,
            rotation_b=0.001 * config.rotaion_b_lr_scale,
            f_dc_b=0.0025 * config.feature_b_lr_scale,
        )
    for group in optimizer.param_groups:
        if group.get("name") in rates:
            group["lr"] = rates[group["name"]]
