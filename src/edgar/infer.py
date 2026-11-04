"""Refine one image or a directory with a fixed subject reference."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--sapiens", required=True, help="Sapiens 0.3B TorchScript .pt2 file"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True, help="New output directory")
    parser.add_argument("--base-model", default="stabilityai/sd-turbo")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument(
        "--preprocessing",
        choices=["training", "application"],
        default="training",
        help="Use application for the historical exp6/v4.1 avatar-refinement convention",
    )
    args = parser.parse_args()

    import torch
    from PIL import Image
    from torchvision.utils import save_image
    from .data import inference_tensor
    from .model import EDGAR

    source = Path(args.input).resolve()
    files = (
        [source]
        if source.is_file()
        else sorted(
            p
            for p in source.rglob("*")
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
    )
    if not files:
        parser.error("No input images found.")
    output = Path(args.output).resolve()
    if output.exists():
        parser.error(
            "Output directory already exists; choose a new directory to avoid overwriting results."
        )
    torch.manual_seed(args.seed)
    model = EDGAR(
        sapiens_path=args.sapiens,
        checkpoint=args.checkpoint,
        base_model=args.base_model,
    )
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    model.to(dtype=dtype).set_eval()
    with Image.open(args.reference) as ref:
        reference = inference_tensor(
            ref, reference=True, preprocessing=args.preprocessing
        )[None].to("cuda", dtype=dtype)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    with torch.inference_mode():
        features = model.encode_reference(reference)
        output.mkdir(parents=True)
        for path in files:
            with Image.open(path) as image:
                tensor = inference_tensor(image, preprocessing=args.preprocessing)[
                    None
                ].to("cuda", dtype=dtype)
            result = model(tensor, reference_features=features, generator=generator)
            relative = Path(path.name) if source.is_file() else path.relative_to(source)
            destination = output / (str(relative) + ".png")
            destination.parent.mkdir(parents=True, exist_ok=True)
            save_image(result[0].float().cpu() * 0.5 + 0.5, destination)
            print(destination, flush=True)


if __name__ == "__main__":
    main()
