"""Coordinate a reviewed-data-gated launch without storing data on the Mac."""
import json
import shlex
import subprocess
import time
from pathlib import Path

from tap import Tap


class Args(Tap):
    run_id: str
    revision: str


def call(argv: list[str]) -> str:
    return subprocess.check_output(argv, text=True).strip()


def remote(rank: int, argv: list[str]) -> str:
    return call(['rx', 'devbox', 'run', 'shi-h200-3', '--server', 'https://relay.radixark.ai',
                 '--rank', str(rank), '--', *argv])


def main(args: Args) -> None:
    cpu = 'ubuntu@100.79.206.53'
    base = '/home/ubuntu/clef-rl-scaled-261009'
    root = '/scratch/' + args.run_id
    while True:
        status = call(['ssh', '-n', cpu, f'test ! -f {base}/publication.exit || cat {base}/publication.exit'])
        if status:
            if status != '0':
                raise RuntimeError('Dataset publication failed; training not launched')
            break
        print('WAITING_FOR_REVIEWED_DATA', flush=True)
        time.sleep(5)
    package = base + '/training-transfer.zip'
    call(['ssh', '-n', cpu, f'cd {base}/data && zip -q {package} train.jsonl validation.jsonl manifest.json validation-report.json publication.json'])
    for rank in (0, 1):
        target = ['ssh'] + (['-F', '/tmp/2dcf7753-shi3-rank1-ssh'] if rank else [])
        target += ['shi-h200-3-sync' if rank else 'shi-h200-3', 'cat > /scratch/2dcf7753-scale-data.zip']
        source = subprocess.Popen(['ssh', cpu, 'cat ' + package], stdout=subprocess.PIPE)
        assert source.stdout is not None
        result = subprocess.run(target, stdin=source.stdout)
        source.stdout.close()
        if source.wait() or result.returncode:
            raise RuntimeError('Streaming dataset transfer failed')
        script = "import zipfile; zipfile.ZipFile('/scratch/2dcf7753-scale-run-source.zip').extractall('/scratch/2dcf7753-scale-prepare')"
        remote(rank, ['/opt/sglang/bin/python', '-c', script])
        remote(rank, ['env', 'PYTHONPATH=/scratch/2dcf7753-scale-prepare', '/opt/sglang/bin/python', '-m',
                      'examples.clef.rl_pilot.prepare_scale', '--run-id', args.run_id, '--node-rank', str(rank),
                      '--source-zip', '/scratch/2dcf7753-scale-run-source.zip', '--source-revision', args.revision,
                      '--pilot-root', '/scratch/261009-a07c2b72', '--data-zip', '/scratch/2dcf7753-scale-data.zip'])
        if remote(rank, ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']):
            raise RuntimeError('GPU processes present; refusing conflicting launch')
        archive = f"from pathlib import Path; import shutil; p=Path('{root}'); d=Path('/global-s3/training-runs/{args.run_id}/provenance/rank{rank}'); d.mkdir(parents=True,exist_ok=True); [(shutil.copyfile(p/n,d/n)) for n in ('manifest.json','launch.fish','source.zip','requirements.freeze.txt')]; print('PROVENANCE_SAVED')"
        remote(rank, ['/opt/sglang/bin/python', '-c', archive])
    for rank in (1, 0):
        name = '2dcf7753-scale-train'
        remote(rank, ['tmux', 'new-session', '-d', '-s', name, 'fish'])
        remote(rank, ['tmux', 'send-keys', '-t', name, 'echo $FISH_VERSION', 'Enter'])
        remote(rank, ['tmux', 'send-keys', '-t', name, 'source ' + root + '/launch.fish', 'Enter'])
    for name, command in (
        ('2dcf7753-scale-logs', 'tail -F ' + root + '/logs/train.log'),
        ('2dcf7753-scale-dashboard', f'env PYTHONPATH={root}/code/miles /opt/sglang/bin/python -m miles.dashboard.serve --dump-details {root}/output --follow --port 7796 > {root}/logs/dashboard.log 2>&1'),
    ):
        remote(0, ['tmux', 'new-session', '-d', '-s', name, 'fish'])
        remote(0, ['tmux', 'send-keys', '-t', name, 'echo $FISH_VERSION', 'Enter'])
        remote(0, ['tmux', 'send-keys', '-t', name, command, 'Enter'])
    print('LAUNCHED', args.run_id, flush=True)


if __name__ == '__main__':
    main(Args(underscores_to_dashes=True).parse_args())
