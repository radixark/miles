"""Collect and analyze finished jobs without moving models or exposing credentials."""

import hashlib
import json
import shutil
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import polars as pl
import wandb
from tap import Tap

from examples.experimental.spec_draft_transition.analyze import main as analyze
from examples.experimental.spec_draft_transition.collect import collect


class Args(Tap):
    root: str
    control_url: str
    manifest: str
    node: int
    data_port: int = 19081


def read_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def update_tracking(root: Path, node: int, jobs: list[dict]) -> None:
    samples = pl.read_parquet(root / f'report/samples-node{node}.parquet')
    errors = []
    for job in jobs:
        if job['slot'] // 8 != node:
            continue
        url = job['result']['wandb_url'].strip()
        base, path = url.split('/radixarkai/', 1)
        parts = path.split('/')
        if base == 'https://forge.coreweave.com/wandb':
            base = 'https://api.forge.coreweave.com'
        try:
            api = wandb.Api(api_key=(root / 'wandb-key').read_text().strip(), overrides={'base_url': base})
            run = api.run(f'radixarkai/{parts[0]}/{parts[-1]}')
            if job['stage'] == 'confirmation':
                for draft in [4, 8, 16]:
                    rows = samples.filter((pl.col('run_id') == job['run_id']) & (pl.col('draft') == draft))
                    for metric in ['accepted', 'emitted', 'corrections', 'bonuses']:
                        run.summary[f'test/k{draft}_{metric}_per_round'] = rows[metric].sum() / rows['blocks'].sum()
            run.summary['validation/k8_expected_prefix'] = job['result']['validation']['expected_prefix']
            run.summary['training_source_revision'] = job['source_revision']
            run.summary['evaluation_source_revision'] = (root / 'repo/SOURCE_REVISION').read_text().strip()
            created = datetime.fromisoformat(run.created_at.replace('Z', '+00:00'))
            day = created.astimezone(ZoneInfo('America/Los_Angeles')).strftime('%y%m%d')
            run.name = f'{day}-{job["run_id"].split("-",1)[1]}-{job["method"]}-t{job["temperature"]}-s{job["seed"]}'
            run.summary.update()
            run.update()
        except (wandb.errors.Error, AttributeError, ValueError, KeyError, TypeError) as error:
            errors.append({'run_id': job['run_id'], 'error': str(error)})
    (root / f'report/tracking-node{node}.json').write_text(json.dumps(errors, indent=2))


def main(args: Args) -> None:
    root = Path(args.root)
    manifest = json.loads(Path(args.manifest).read_text())
    while True:
        state = read_json(args.control_url + '/state')
        if state['phase'] == 'complete':
            break
        print(json.dumps({'phase': state['phase'], 'complete': sum(j['status'] == 'complete' for j in state['jobs'])}), flush=True)
        time.sleep(15)
    jobs = [job for job in state['jobs'] if job['stage'] == 'confirmation']
    expected = sum(job['stage'] == 'confirmation' for job in manifest['jobs'])
    assert len(jobs) == expected and all(job['status'] == 'complete' for job in jobs)
    identity = hashlib.sha256(json.dumps(sorted(job['run_id'] for job in jobs)).encode()).hexdigest()
    for job in jobs:
        if job['slot'] // 8 != args.node:
            continue
        config = json.loads((Path(job['result']['output']) / 'config.json').read_text())
        selected = state['selected_rates'][f'{job["temperature"]}/{job["method"]}']
        assert not selected['boundary']
        assert config['learning_rate'] == job['learning_rate'] == selected['rate']
    collect(SimpleNamespace(root=args.root, control_url=args.control_url, node=args.node))
    transfer = root / 'data/report'
    transfer.mkdir(parents=True, exist_ok=True)
    for name in ['samples', 'curves', 'jobs']:
        filename = f'{name}-node{args.node}.parquet'
        shutil.copyfile(root / 'report' / filename, transfer / filename)
    (transfer / f'collection-node{args.node}.json').write_text(json.dumps({'identity': identity}))
    print('COLLECTION_COMPLETE', args.node, flush=True)
    if args.node != 0:
        update_tracking(root, args.node, state['jobs'])
        return
    for node, host in enumerate(manifest['nodes']):
        if node == 0:
            continue
        base = f'http://{host}:{args.data_port}/report/'
        while True:
            try:
                marker = read_json(base + f'collection-node{node}.json')
            except (urllib.error.URLError, TimeoutError):
                time.sleep(15)
                continue
            assert marker['identity'] == identity
            break
        for name in ['samples', 'curves', 'jobs']:
            filename = f'{name}-node{node}.parquet'
            with urllib.request.urlopen(base + filename, timeout=60) as response, (root / 'report' / filename).open('wb') as output:
                shutil.copyfileobj(response, output)
    analyze(SimpleNamespace(root=str(root / 'report'), resamples=20000, seed=20261108,
                            baseline_root=manifest.get('baseline_root', '')))
    (root / 'report/ANALYSIS_COMPLETE').write_text(identity)
    print('ANALYSIS_COMPLETE', identity, flush=True)
    update_tracking(root, args.node, state['jobs'])


if __name__ == '__main__':
    main(Args().parse_args())
