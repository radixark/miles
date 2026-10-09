"""Stream and checksum a complete HF checkpoint from an object-store mount."""

import hashlib
import json
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tap import Tap


class Args(Tap):
    checkpoint: Path
    output: Path
    workers: int = 4


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stage_file(checkpoint: Path, output: Path, entry: dict) -> dict:
    relative = Path(entry["path"])
    if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "hf":
        raise ValueError("invalid export path")
    destination = output / relative.relative_to("hf")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == entry["bytes"] and digest_file(destination) == entry["sha256"]:
        return entry
    temporary = destination.with_suffix(destination.suffix + ".partial")
    digest = hashlib.sha256()
    size = 0
    with (checkpoint / relative).open("rb") as source, temporary.open("wb") as target:
        while chunk := source.read(8 * 1024 * 1024):
            target.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    if size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
        raise ValueError(f"export checksum mismatch: {relative}")
    temporary.replace(destination)
    return entry


def main() -> None:
    args = Args().parse_args()
    if args.workers < 1 or not (args.checkpoint / "COMPLETE.json").is_file():
        raise ValueError("invalid worker count or incomplete checkpoint")
    manifest = args.checkpoint / "upload-manifest.json"
    files = [entry for entry in json.loads(manifest.read_text())["files"] if entry["path"].startswith("hf/")]
    names = {entry["path"] for entry in files}
    if len(names) != len(files) or not {"hf/config.json", "hf/joint_head.safetensors", "hf/joint_head_config.json", "hf/model.safetensors.index.json"} <= names:
        raise ValueError("incomplete or duplicated HF manifest")
    index = json.loads((args.checkpoint / "hf" / "model.safetensors.index.json").read_text())
    if not {"hf/" + value for value in index["weight_map"].values()} <= names:
        raise ValueError("manifest omits backbone shards")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(stage_file, args.checkpoint, args.output, entry) for entry in files]
        for count, future in enumerate(as_completed(futures), 1):
            entry = future.result()
            print("VERIFIED", count, len(files), entry["path"], entry["bytes"], flush=True)
    receipt = {"host": socket.gethostname(), "checkpoint": str(args.checkpoint), "manifest_sha256": digest_file(manifest), "files": files}
    (args.output / "STAGED.json").write_text(json.dumps(receipt, indent=2))
    print("STAGE_COMPLETE", len(files), flush=True)


if __name__ == "__main__":
    main()
