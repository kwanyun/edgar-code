"""Verify EDGAR-SET shard checksums and extract into a new training directory."""

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unpack(package, output):
    package, output = Path(package).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    shards = json.loads((package / "shards.json").read_text())
    if not shards:
        raise ValueError("No dataset shards listed.")
    # Verify all archive hashes before writing extracted data.
    for record in shards:
        name = record["file"]
        if Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError("Invalid shard filename.")
        path = package / "shards" / name
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Corrupt or incomplete archive: {name}")
        print(f"Verified {name}", flush=True)
    output.mkdir(parents=True)
    seen = set()
    for record in shards:
        with tarfile.open(package / "shards" / record["file"], "r:") as archive:
            count = 0
            for member in archive:
                relative = PurePosixPath(member.name)
                if (
                    not member.isfile()
                    or relative.is_absolute()
                    or ".." in relative.parts
                    or "\\" in member.name
                    or ":" in member.name
                    or relative.parts[0] != "triplets"
                ):
                    raise ValueError(f"Invalid archive member: {member.name}")
                if member.name in seen:
                    raise ValueError(f"Duplicate archive member: {member.name}")
                seen.add(member.name)
                destination = output.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with (
                    archive.extractfile(member) as source,
                    destination.open("xb") as target,
                ):
                    shutil.copyfileobj(source, target)
                count += 1
            if count != record["samples"]:
                raise ValueError(f"Archive sample count mismatch: {record['file']}")
        print(f"Extracted {record['file']}", flush=True)
    for name in (
        "manifest.json",
        "manifest_identity_disjoint.json",
        "index.jsonl",
        "README.md",
        "LICENSE-CAFCA.txt",
        "NOTICE.md",
        "release_status.json",
    ):
        source = package / name
        if source.is_file():
            shutil.copy2(source, output / name)
    return len(seen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(f"Extracted {unpack(args.package, args.output)} triplets.")
