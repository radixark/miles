"""Prepare a matched calibration control from a completed pilot, on each node."""
import hashlib
import json
import shutil
import socket
import zipfile
from pathlib import Path

from tap import Tap


class Args(Tap):
    run_id: str
    node_rank: int
    source_zip: Path
    source_revision: str
    pilot_root: Path


def main(args: Args) -> None:
    root = Path("/scratch") / args.run_id
    root.mkdir(exist_ok=False)
    for name in ("logs", "code"):
        (root / name).mkdir()
    code = root / "code" / "miles"
    zipfile.ZipFile(args.source_zip).extractall(code)
    shutil.copyfile(args.source_zip, root / "source.zip")
    for name in ("model", "data", "forecastbench"):
        (root / name).symlink_to(args.pilot_root / name, target_is_directory=True)
    old_id = args.pilot_root.name
    launch = (args.pilot_root / "launch.fish").read_text().replace(old_id, args.run_id)
    launch = launch.replace("261009-clef-rl-hard2048-g32-a07c2b72", "261009-clef-brier-control-" + args.run_id.split("-")[-1])
    launch = launch.replace("--brier-weight 1.0", "--policy-weight 0 --brier-weight 1.0")
    launch = launch.replace("--master-port=29694", "--master-port=29695").replace("--prometheus-port 9094", "--prometheus-port 9095")
    (root / "launch.fish").write_text(launch)
    manifest = json.loads((args.pilot_root / "manifest.json").read_text().replace(old_id, args.run_id))
    manifest.update({
        "run_id": args.run_id, "status": "prepared", "node_rank": args.node_rank, "host": socket.gethostname(),
        "training": "Matched Brier-plus-KL supervised control; categorical sampling retained, policy gradient weight zero",
        "launch_command": launch.split(" > ")[0], "control_of": old_id,
        "source_revision": args.source_revision, "source_archive_sha256": hashlib.sha256(args.source_zip.read_bytes()).hexdigest(),
        "intent": "Compare 64 updates against completed hybrid pilot on identical 2048train/256validation, seed and initialization",
    })
    for stale in ("exit_code", "final_checkpoint_complete", "wandb_url"):
        manifest.pop(stale, None)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    shutil.copyfile(args.pilot_root / "requirements.freeze.txt", root / "requirements.freeze.txt")
    for name in ("compare.py", "finalize.py"):
        text = (args.pilot_root / name).read_text().replace(old_id, args.run_id)
        text = text.replace("https://wandb.ai/radixarkai/clef-model-training/runs/2ctt9x6c", "")
        if name == "finalize.py":
            text = text.replace('"wandb_url":""', '"wandb_url":(root/"output/wandb-url.txt").read_text().strip()')
        (root / name).write_text(text)
    print("CONTROL_PREPARED", root, flush=True)


if __name__ == "__main__":
    main(Args(underscores_to_dashes=True).parse_args())

