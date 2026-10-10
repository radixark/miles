"""Prepare a full-size hybrid run using a reviewed, published dataset."""
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
    data_zip: Path


def main(args: Args) -> None:
    root = Path('/scratch') / args.run_id
    root.mkdir(exist_ok=False)
    (root / 'logs').mkdir()
    code = root / 'code/miles'
    code.mkdir(parents=True)
    zipfile.ZipFile(args.source_zip).extractall(code)
    shutil.copyfile(args.source_zip, root / 'source.zip')
    data = root / 'data'
    data.mkdir()
    zipfile.ZipFile(args.data_zip).extractall(data)
    manifest = json.loads((data / 'manifest.json').read_text())
    receipt = json.loads((data / 'publication.json').read_text())
    if not 32000 <= manifest['train'] <= 32768 or not 1800 <= manifest['validation'] <= 2048:
        raise ValueError('Unexpected dataset size')
    for name in ('train.jsonl', 'validation.jsonl', 'manifest.json', 'validation-report.json'):
        if hashlib.sha256((data / name).read_bytes()).hexdigest() != receipt['objects'][name]['sha256']:
            raise ValueError(f'Dataset checksum mismatch: {name}')
    for name in ('model', 'forecastbench'):
        (root / name).symlink_to(args.pilot_root / name, target_is_directory=True)
    old_id = args.pilot_root.name
    launch = (args.pilot_root / 'launch.fish').read_text().replace(old_id, args.run_id)
    launch = launch.replace('261009-clef-rl-hard2048-g32-a07c2b72', args.run_id + '-clef-rl-scaled32768-g32')
    steps = 2 * (manifest['train'] // 64)
    for old, new in (('--max-steps 64', f'--max-steps {steps}'), ('--save-interval 32', '--save-interval 128'),
                     ('--eval-interval 16', '--eval-interval 32'), ('--master-port=29694', '--master-port=29696'),
                     ('--prometheus-port 9094', '--prometheus-port 9096')):
        launch = launch.replace(old, new)
    (root / 'launch.fish').write_text(launch)
    run = json.loads((args.pilot_root / 'manifest.json').read_text().replace(old_id, args.run_id))
    for name in ('exit_code', 'final_checkpoint_complete', 'wandb_url'):
        run.pop(name, None)
    run.update(run_id=args.run_id, status='prepared', host=socket.gethostname(), node_rank=args.node_rank,
               source_revision=args.source_revision, source_archive_sha256=hashlib.sha256(args.source_zip.read_bytes()).hexdigest(),
               launch_command=launch.split(' > ')[0], dataset=manifest, publication=receipt,
               training=f'Full-parameter hybrid categorical GRPO plus Brier and KL; {steps} updates',
               initialization='Original supervised step2048, fresh optimizer; not pilot weights')
    (root / 'manifest.json').write_text(json.dumps(run, indent=2))
    shutil.copyfile(args.pilot_root / 'requirements.freeze.txt', root / 'requirements.freeze.txt')
    print('PREPARED', root, flush=True)


if __name__ == '__main__':
    main(Args(underscores_to_dashes=True).parse_args())
