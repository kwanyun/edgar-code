"""Native-avatar adapter -> EDGAR pseudo-targets -> shared-avatar optimization."""

import argparse
import importlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    integration = p.add_mutually_exclusive_group(required=True)
    integration.add_argument(
        "--pipeline",
        choices=["rgbavatar", "cap4d"],
        help="Built-in native pipeline adapter",
    )
    integration.add_argument(
        "--adapter", help="Custom module:factory, returning avatar, views, optimizer"
    )
    p.add_argument("--config", required=True, help="Adapter configuration JSON")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--sapiens", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument(
        "--preprocessing",
        choices=["training", "application"],
        default="application",
        help="Reference convention; use training for weights trained with this package",
    )
    args = p.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        p.error("Output already exists; choose a new directory.")
    if args.steps < 1:
        p.error("--steps must be positive.")
    import torch
    from PIL import Image
    from .model import EDGAR
    from .data import inference_tensor
    from .avatar import generate_targets, optimize_avatar

    torch.manual_seed(args.seed)
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    # Resolve paths before entering the native repository's working directory.
    if args.pipeline:
        for key in (
            "repository",
            "native_config",
            "baseline",
            "transforms",
            "flame_model",
        ):
            if key in config:
                config[key] = str((config_path.parent / config[key]).resolve())
        if "source_paths" in config:
            config["source_paths"] = [
                str((config_path.parent / path).resolve())
                for path in config["source_paths"]
            ]
    checkpoint, sapiens, reference_path = [
        Path(value).resolve()
        for value in (args.checkpoint, args.sapiens, args.reference)
    ]
    for path in (checkpoint, sapiens, reference_path):
        if not path.is_file():
            p.error(f"File does not exist: {path}")
    if args.pipeline:
        config["seed"] = args.seed
        config["_workspace"] = str(output / "native")
    module, factory = (
        (f"edgar.adapters.{args.pipeline}", "create")
        if args.pipeline
        else args.adapter.rsplit(":", 1)
    )
    output.mkdir(parents=True)
    avatar, views, optimizer = getattr(importlib.import_module(module), factory)(config)
    model = EDGAR(sapiens_path=str(sapiens), checkpoint=str(checkpoint))
    model.set_eval()
    with Image.open(reference_path) as f:
        reference = inference_tensor(
            f, reference=True, preprocessing=args.preprocessing
        )[None]
    targets = generate_targets(model, avatar, views, reference, args.seed)
    del model
    torch.cuda.empty_cache()
    losses = optimize_avatar(avatar, views, targets, optimizer, args.steps, args.seed)
    avatar.save(output / "avatar")
    (output / "refinement.json").write_text(
        json.dumps(
            {
                "steps": args.steps,
                "views": len(views),
                "gaussians": avatar.gaussian_count,
                "seed": args.seed,
                "preprocessing": args.preprocessing,
                "pipeline": args.pipeline or args.adapter,
                "sampled_views": views if args.pipeline else None,
                "config": config,
                "checkpoint": str(checkpoint),
                "reference": str(reference_path),
                "losses": losses,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
