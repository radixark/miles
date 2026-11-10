"""Collect compact metrics on each devbox node without moving checkpoints."""

import json
import urllib.request
from pathlib import Path

import polars as pl
from tap import Tap


class Args(Tap):
    root: str
    control_url: str
    node: int


def collect(args: Args) -> None:
    root=Path(args.root)
    with urllib.request.urlopen(args.control_url+'/state',timeout=30) as response:
        state=json.load(response)
    assert state['phase']=='complete',state['phase']
    samples,curves,jobs=[],[],[]
    for job in state['jobs']:
        if job['stage']!='confirmation' or job['slot']//8!=args.node:
            continue
        assert job['status']=='complete'
        directory=Path(job['result']['output'])
        common={key:job[key] for key in ['run_id','temperature','method','seed','learning_rate']}
        jobs.append({**common,**json.loads((directory/'complete.json').read_text()),
                     'wandb_url':job['result']['wandb_url'],'node':args.node})
        for line in (directory/'metrics.jsonl').read_text().splitlines():
            metric=json.loads(line)
            if metric['kind']=='validation':
                curves.append({**common,**metric})
        for draft,expected_rows in [(8,2000),(4,256),(16,256)]:
            lines=(directory/f'test-k{draft}.jsonl').read_text().splitlines()
            assert len(lines)==expected_rows,(job['run_id'],draft,len(lines))
            identities=set()
            for line in lines:
                row=json.loads(line)
                assert row['prompt_id'] not in identities
                identities.add(row['prompt_id'])
                assert len(row['accepted'])==len(row['emitted']) and len(row['accepted'])>0
                assert all(0<=value<=draft for value in row['accepted'])
                assert len(row['generated_ids'])==sum(row['emitted'])
                samples.append({**common,'draft':draft,'prompt_id':row['prompt_id'],
                    'accepted':sum(row['accepted']),'emitted':sum(row['emitted']),
                    'blocks':len(row['accepted']),'corrections':row['corrections'],'bonuses':row['bonuses']})
    destination=root/'report'
    destination.mkdir(exist_ok=True)
    pl.DataFrame(samples).write_parquet(destination/f'samples-node{args.node}.parquet')
    pl.DataFrame(curves).write_parquet(destination/f'curves-node{args.node}.parquet')
    pl.DataFrame(jobs).write_parquet(destination/f'jobs-node{args.node}.parquet')
    (destination/'state.json').write_text(json.dumps(state,indent=2))
    print(json.dumps({'node':args.node,'models':len(jobs),'samples':len(samples),'curves':len(curves)}))


if __name__=='__main__':
    collect(Args().parse_args())
