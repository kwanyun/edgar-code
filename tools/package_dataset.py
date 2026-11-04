"""Package recovered EDGAR triplets without modifying the source dataset.

The NeRSemble package is private staging until redistribution permission is obtained.
Run with Pillow installed; PyTorch is not needed.
"""

import argparse
import hashlib
import io
import json
import os
import re
import tarfile
from pathlib import Path
from PIL import Image


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect(root, source):
    records = []
    folder = root / ("GS" if source == "CAFCA" else "RGB")
    for parent, dirs, files in os.walk(folder):
        dirs[:] = sorted(
            d
            for d in dirs
            if not d.startswith(".") and not (Path(parent) / d).is_symlink()
        )
        for name in sorted(files):
            path = Path(parent) / name
            if (
                path.suffix.lower() not in {".png", ".jpg", ".jpeg"}
                or path.is_symlink()
            ):
                continue
            relative = path.relative_to(folder)
            identity = relative.parts[1] if source == "CAFCA" else relative.parts[0]
            pattern = r"\d{5}" if source == "CAFCA" else r"\d{3}"
            if not re.fullmatch(pattern, identity):
                raise ValueError(f"Unrecognized identity path: {relative}")
            records.append((path, f"triplets/{source}/{relative.as_posix()}", identity))
    if not records:
        raise ValueError(f"No images found in {folder}")
    return sorted(records, key=lambda row: row[1])


def package_source(root, destination, source, shard_bytes):
    destination.mkdir(parents=True, exist_ok=False)
    records = collect(root, source)
    index = []
    shards = []
    archive = None
    shard_path = None
    current_bytes = 0
    try:
        for number, (path, relative, identity) in enumerate(records):
            content = path.read_bytes()
            with Image.open(io.BytesIO(content)) as image:
                if image.size != (1536, 512) or image.mode not in {"RGB", "RGBA"}:
                    raise ValueError(
                        f"Expected 1536x512 RGB/RGBA triplet: {path} ({image.size}, {image.mode})"
                    )
                image.verify()
            if archive is None or current_bytes + len(content) > shard_bytes:
                if archive is not None:
                    archive.close()
                    shards[-1]["bytes"] = shard_path.stat().st_size
                    shards[-1]["sha256"] = digest_file(shard_path)
                shard_path = destination / f"{source.lower()}-{len(shards):05d}.tar"
                archive = tarfile.open(shard_path, "w", format=tarfile.PAX_FORMAT)
                shards.append({"file": shard_path.name, "samples": 0})
                current_bytes = 0
            info = tarfile.TarInfo(relative)
            info.size = len(content)
            info.mode = 0o644
            info.mtime = 0
            archive.addfile(info, io.BytesIO(content))
            current_bytes += len(content)
            shards[-1]["samples"] += 1
            index.append(
                {
                    "triplet": relative,
                    "source": source,
                    "identity": identity,
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "shard": shard_path.name,
                }
            )
            if (number + 1) % 5000 == 0:
                print(
                    f"{source}: {number + 1}/{len(records)} verified and packed",
                    flush=True,
                )
    finally:
        if archive is not None:
            archive.close()
    shards[-1]["bytes"] = shard_path.stat().st_size
    shards[-1]["sha256"] = digest_file(shard_path)
    (destination / "index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in index)
    )
    (destination / "shards.json").write_text(json.dumps(shards, indent=2) + "\n")
    return index, shards


def write_manifest(path, records, identity_split=False):
    if identity_split:
        held_out = set()
        for source in sorted({r["source"] for r in records}):
            identities = sorted(
                {r["identity"] for r in records if r["source"] == source},
                key=lambda value: hashlib.sha256(
                    f"2026:{source}:{value}".encode()
                ).hexdigest(),
            )
            held_out.update(
                (source, identity)
                for identity in identities[: max(1, round(len(identities) * 0.02))]
            )
        train = [r for r in records if (r["source"], r["identity"]) not in held_out]
        validation = [r for r in records if (r["source"], r["identity"]) in held_out]
        policy = "Release-defined identity-disjoint 98/2 split, SHA256 ordering with seed 2026; not the original paper split."
    else:
        cutoff = int(len(records) * 0.99)
        train, validation = records[:cutoff], records[cutoff:]
        policy = "Recovered loader policy on canonical images (editor backups excluded): sorted paths, first 99% train and last 1% validation. May share identities."
    simplify = lambda values: [
        {k: r[k] for k in ("triplet", "source", "identity")} for r in values
    ]
    data = {
        "split_policy": policy,
        "panel_order": ["reference", "target", "input"],
        "train": simplify(train),
        "validation": simplify(validation),
    }
    path.write_text(json.dumps(data, separators=(",", ":")) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Recovered dataset root containing GS/ and RGB/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New private release-staging directory",
    )
    parser.add_argument("--shard-mib", type=int, default=1024)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new directory.")
    if args.shard_mib < 1:
        parser.error("--shard-mib must be positive.")
    args.output.mkdir(parents=True)
    all_records = []
    all_shards = []
    summary = {}
    for source in ("CAFCA", "NeRSemble"):
        records, shards = package_source(
            args.root,
            args.output / "components" / source,
            source,
            args.shard_mib * 1024**2,
        )
        all_records.extend(records)
        all_shards.extend(shards)
        summary[source] = {
            "triplets": len(records),
            "identities": len({r["identity"] for r in records}),
            "image_bytes": sum(r["bytes"] for r in records),
            "archive_bytes": sum(r["bytes"] for r in shards),
            "shards": len(shards),
        }
        print(json.dumps({source: summary[source]}), flush=True)
    for name, records, sources in [
        ("CAFCA-only", [r for r in all_records if r["source"] == "CAFCA"], ["CAFCA"]),
        ("Both", all_records, ["CAFCA", "NeRSemble"]),
    ]:
        folder = args.output / name
        (folder / "shards").mkdir(parents=True)
        selected_shards = []
        for source in sources:
            for shard in (args.output / "components" / source).glob("*.tar"):
                os.link(shard, folder / "shards" / shard.name)
            selected_shards.extend(
                json.loads(
                    (args.output / "components" / source / "shards.json").read_text()
                )
            )
        write_manifest(folder / "manifest.json", records)
        write_manifest(
            folder / "manifest_identity_disjoint.json", records, identity_split=True
        )
        (folder / "index.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in records)
        )
        (folder / "shards.json").write_text(
            json.dumps(selected_shards, indent=2) + "\n"
        )
        (folder / "SHA256SUMS").write_text(
            "".join(f"{s['sha256']}  shards/{s['file']}\n" for s in selected_shards)
        )
        (folder / "release_status.json").write_text(
            json.dumps(
                {
                    "package": name,
                    "status": "private-pending-NeRSemble-redistribution-permission"
                    if name == "Both"
                    else "CAFCA-derived-CC-BY-NC-SA-4.0",
                    "panel_order": ["reference", "target", "input"],
                    "image_size": [1536, 512],
                    "sources": {source: summary[source] for source in sources},
                },
                indent=2,
            )
            + "\n"
        )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Dataset packages complete.", flush=True)


if __name__ == "__main__":
    main()
