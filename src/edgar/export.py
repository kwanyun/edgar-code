"""Export inference-only EDGAR weights without training state."""

import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    output = Path(a.output)
    if output.exists():
        p.error("Output already exists.")
    import torch
    from .model import read_checkpoint

    state = read_checkpoint(a.checkpoint)
    allowed = {
        "format_version",
        "model_config",
        "rank_vae",
        "vae_lora_target_modules",
        "state_dict_unet",
        "state_dict_vae",
        "state_dict_ref_proj",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v for k, v in state.items() if k in allowed}, output)
    print(output)


if __name__ == "__main__":
    main()
