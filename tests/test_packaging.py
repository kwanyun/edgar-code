"""Exercise real archive and split behavior with non-photographic fixtures."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from PIL import Image


def load_tool(name):
    path = Path(__file__).resolve().parents[1] / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dataset_packages_round_trip_and_split(tmp_path, monkeypatch):
    package = load_tool("package_dataset")
    unpack = load_tool("unpack_dataset")
    source = tmp_path / "source"
    for folder in (
        "GS/00000-00099/00000",
        "GS/00000-00099/00001",
        "RGB/001/16_1/000",
        "RGB/002/16_1/000",
    ):
        path = source / folder
        path.mkdir(parents=True)
        for index in range(2):
            image = Image.new("RGB", (1536, 512), (index * 20, 90, 120))
            image.save(path / f"{index:05d}.png")
    destination = tmp_path / "packaged"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "package_dataset",
            "--root",
            str(source),
            "--output",
            str(destination),
            "--shard-mib",
            "1",
        ],
    )
    package.main()
    assert unpack.unpack(destination / "Both", tmp_path / "unpacked") == 8
    manifest = json.loads(
        (destination / "Both" / "manifest_identity_disjoint.json").read_text()
    )
    identities = lambda split: {
        (row["source"], row["identity"]) for row in manifest[split]
    }
    assert identities("train").isdisjoint(identities("validation"))
    assert (
        len(
            json.loads((destination / "CAFCA-only" / "manifest.json").read_text())[
                "train"
            ]
        )
        == 3
    )
    shard = next((destination / "CAFCA-only" / "shards").glob("*.tar"))
    with shard.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="Corrupt"):
        unpack.unpack(destination / "CAFCA-only", tmp_path / "bad")
