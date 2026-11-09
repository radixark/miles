"""Extend boundary searches using run IDs and snapshots reserved in advance.

The devbox needs no GitHub credentials or laptop connection. Only reserve entries
whose complete code snapshots were archived before startup may be scheduled.
"""

import json
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from tap import Tap


class Args(Tap):
    root: str
    control_url: str
    reserve: str


def request(url: str, payload: dict | None = None) -> dict:
    data=None if payload is None else json.dumps(payload).encode()
    headers={'Content-Type':'application/json'}
    with urllib.request.urlopen(urllib.request.Request(url,data=data,headers=headers),timeout=30) as response:
        return json.load(response)


def extend_jobs(state: dict, reserve: dict) -> list[dict]:
    extra=[]
    for key,choice in state['selected_rates'].items():
        if not choice['boundary']:
            continue
        temperature,method=key.split('/')
        group=[job for job in state['jobs'] if job['stage']=='tuning' and job['temperature']==float(temperature) and job['method']==method]
        if not all(job['status']=='complete' for job in group):
            continue
        rates=sorted({job['learning_rate'] for job in group})
        if choice['rate']==min(rates):
            candidates=[rate for rate in reserve['grid'] if rate<min(rates)]
            new=max(candidates) if candidates else None
        else:
            candidates=[rate for rate in reserve['grid'] if rate>max(rates)]
            new=min(candidates) if candidates else None
        if new is None:
            raise RuntimeError(f'Grid exhausted for {key}: {rates}')
        for seed in sorted({job['seed'] for job in group}):
            matching=[entry for entry in reserve['jobs'] if entry['temperature']==float(temperature)
                      and entry['method']==method and entry['learning_rate']==new and entry['seed']==seed]
            assert len(matching)==1,(key,new,seed)
            assert matching[0]['run_id'] not in {job['run_id'] for job in state['jobs']}
            extra.append({**matching[0],'status':'pending'})
    return extra


def main(args: Args) -> None:
    root=Path(args.root)
    reserve=json.loads(Path(args.reserve).read_text())
    assert reserve['archive_verified']
    previous,error_message=None,None
    while True:
        try:
            state=request(args.control_url+'/state')
            assert reserve['training_revision'].startswith(state['source_revision'])
            counts={status:sum(job['status']==status for job in state['jobs']) for status in ['pending','running','complete','failed']}
            summary=(state['phase'],tuple(counts.items()))
            if summary!=previous:
                print(datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),state['phase'],counts,flush=True)
                previous=summary
            if state['phase']=='complete':
                (root/'control/all-jobs-complete.json').write_text(json.dumps({'jobs':len(state['jobs'])}))
                print('ALL_EXPERIMENT_JOBS_COMPLETE',flush=True)
                break
            if state['phase'] not in ['failure_attention','generation']:
                jobs=extend_jobs(state,reserve)
                if jobs:
                    with (root/'control/rate-extensions.jsonl').open('a') as handle:
                        handle.write(json.dumps({'source_revision':reserve['training_revision'],'jobs':jobs})+'\n')
                    request(args.control_url+'/extend',{'jobs':jobs})
                    print('EXTENDED_GRID',[(job['temperature'],job['method'],job['learning_rate'],job['run_id']) for job in jobs],flush=True)
            error_message=None
        except (urllib.error.URLError,ValueError,TimeoutError,RuntimeError) as error:
            if str(error)!=error_message:
                print('WATCHER_ERROR',str(error),flush=True)
                error_message=str(error)
        time.sleep(15)


if __name__=='__main__':
    main(Args().parse_args())
