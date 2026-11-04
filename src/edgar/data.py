"""Explicit triplet manifests: reference, ground truth, degraded rendering."""

import json
from pathlib import Path
import random

import torch
from PIL import Image
from torchvision.transforms import functional as F
from torch.nn import functional as torch_f

PROMPT = "High resolution portrait"
MEAN = [123.5, 116.5, 103.5]
STD = [58.5, 57.0, 57.5]


def image_tensor(image, resolution=512, reference=False):
    image = F.to_tensor(image.convert("RGB"))
    image = F.resize(
        image, [1024, 1024] if reference else [resolution, resolution], antialias=True
    )
    return F.normalize(image * 255, MEAN, STD) if reference else image * 2 - 1


def inference_tensor(image, reference=False, preprocessing="training"):
    """Keep the training and historical avatar-application conventions explicit."""
    if preprocessing == "training":
        return image_tensor(image, reference=reference)
    if preprocessing != "application":
        raise ValueError(f"Unknown preprocessing convention: {preprocessing}")
    image = image.convert("RGB")
    if reference:
        # Application scripts resize the PIL image before converting to [-1,1].
        return F.to_tensor(F.resize(image, [1024, 1024])) * 2 - 1
    tensor = F.to_tensor(image) * 2 - 1
    _, height, width = tensor.shape
    pad_h, pad_w = max(height, width) - height, max(height, width) - width
    tensor = torch_f.pad(
        tensor,
        (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
        value=1,
    )
    return torch_f.interpolate(
        tensor[None], size=(512, 512), mode="bilinear", align_corners=False
    )[0]


def read_manifest(path, split):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if split not in data or not data[split]:
        raise ValueError(f"Manifest requires a nonempty '{split}' list.")
    records = []
    for item in data[split]:
        keys = ("triplet",) if "triplet" in item else ("input", "target", "reference")
        record = {}
        for key in keys:
            p = (path.parent / item[key]).resolve()
            if not p.is_file():
                raise FileNotFoundError(p)
            record[key] = p
        records.append(record)
    return records


class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, manifest, split, resolution=512):
        if resolution != 512:
            raise ValueError(
                "Training uses 512x512 images and fixed spatial loss crops."
            )
        self.records = read_manifest(manifest, split)
        self.augment = split == "train"
        self.resolution = resolution

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        if "triplet" in record:
            with Image.open(record["triplet"]) as f:
                packed = f.convert("RGB")
            if packed.size != (1536, 512):
                raise ValueError(
                    f"Expected a 1536x512 [reference | target | input] triplet: {record['triplet']}"
                )
            reference, target, source = [
                packed.crop((i * 512, 0, (i + 1) * 512, 512)) for i in range(3)
            ]
        else:
            images = []
            for key in ("input", "target", "reference"):
                with Image.open(record[key]) as f:
                    images.append(f.convert("RGB"))
            source, target, reference = images
            if source.size != target.size:
                raise ValueError(
                    "Input and target must have matching spatial dimensions."
                )
        if self.augment:
            angle = random.uniform(-10, 10)
            source, target = [F.rotate(x, angle, fill=255) for x in (source, target)]
            scale = 1 + random.uniform(-0.3, 0.1)
            h, w = source.height, source.width
            h, w = (
                int(h * (scale + random.uniform(-0.02, 0.02))),
                int(w * (scale + random.uniform(-0.02, 0.02))),
            )
            source, target = [F.resize(x, [h, w]) for x in (source, target)]
            size = int(min(h, w) * (1 - random.uniform(0, 0.1)))
            top, left = random.randint(0, h - size), random.randint(0, w - size)
            source, target = [
                F.crop(x, top, left, size, size) for x in (source, target)
            ]
            brightness, saturation = (
                1 + random.uniform(-0.1, 0.1),
                1 + random.uniform(-0.1, 0.1),
            )
            source, target, reference = [
                F.adjust_saturation(F.adjust_brightness(x, brightness), saturation)
                for x in (source, target, reference)
            ]
        return {
            "input": image_tensor(source),
            "target": image_tensor(target),
            "reference": image_tensor(reference, reference=True),
        }
