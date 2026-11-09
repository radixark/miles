"""Extend boundary learning-rate searches after archiving each new run.

The archive already contains the frozen training/library commits. Git over
HTTPS and a protected credential helper let this controller operate on the
devbox without a persistent laptop connection or a copied SSH key.
"""

import json
import os
import secrets
import subprocess
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
    token_file: str
    archive_repo: str
    training_revision: str
    grid: list[float] = [1e-7,3e-7,1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3]


def request(url: str, payload: dict | None = None, token: str | None = None) -> dict:
    headers={'Accept':'application/vnd.github+json','Content-Type':'application/json'}
    if token:
        headers['Authorization']='Bearer '+token
        headers['X-GitHub-Api-Version']='2022-11-28'
    data=None if payload is None else json.dumps(payload).encode()
    with urllib.request.urlopen(urllib.request.Request(url,data=data,headers=headers),timeout=30) as response:
        return json.load(response)


def extend_jobs(state: dict, grid: list[float]) -> list[dict]:
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
            candidates=[rate for rate in grid if rate<min(rates)]
            new=max(candidates) if candidates else None
        else:
            candidates=[rate for rate in grid if rate>max(rates)]
            new=min(candidates) if candidates else None
        if new is None:
            print('GRID_LIMIT_ATTENTION',key,choice,flush=True)
            continue
        for seed in sorted({job['seed'] for job in group}):
            day=datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%y%m%d')
            extra.append({'run_id':day+'-'+secrets.token_hex(4),'stage':'tuning','temperature':float(temperature),
                          'method':method,'learning_rate':new,'seed':seed,'updates':2000,'status':'pending'})
    return extra


def archive_jobs(args: Args, jobs: list[dict], token: str, manifest: dict) -> None:
    versions={'miles':args.training_revision,'sglang':manifest['sglang'],'megatron':manifest['megatron']}
    root=Path(args.root)
    store=root/'snapshot-objects.git'
    askpass=root/'snapshot-askpass'
    askpass.write_text('#!/usr/bin/env python3\nimport pathlib,sys\nprint("x-access-token" if "Username" in sys.argv[1] else pathlib.Path('+repr(args.token_file)+').read_text().strip())\n')
    askpass.chmod(0o700)
    environment={**os.environ,'GIT_ASKPASS':str(askpass),'GIT_TERMINAL_PROMPT':'0',
                 'GIT_CONFIG_GLOBAL':'/dev/null','GIT_CONFIG_SYSTEM':'/dev/null',
                 'GIT_TRACE':'0','GIT_TRACE_CURL':'0','GIT_CURL_VERBOSE':'0'}
    url=f'https://github.com/{args.archive_repo}.git'
    if not store.exists():
        subprocess.run(['git','init','--bare',str(store)],check=True,env=environment)
        subprocess.run(['git','-C',str(store),'config','user.name','Shi Dong'],check=True)
        subprocess.run(['git','-C',str(store),'config','user.email','shi.dong@radixark.ai'],check=True)
    for component,revision in versions.items():
        if subprocess.run(['git','-C',str(store),'cat-file','-e',revision],capture_output=True).returncode:
            reference=f"refs/training-runs/{manifest['experiment_id']}/{component}"
            subprocess.run(['git','-c','credential.helper=','-C',str(store),'fetch','--depth=1',url,reference+':refs/base/'+component],check=True,env=environment)
    refs=[f"{revision}:refs/training-runs/{job['run_id']}/{component}" for job in jobs for component,revision in versions.items()]
    subprocess.run(['git','-c','credential.helper=','-C',str(store),'push',url,*refs],check=True,env=environment)


def main(args: Args) -> None:
    root=Path(args.root)
    token=Path(args.token_file).read_text().strip()
    manifest=json.loads((root/'repo/examples/experimental/spec_draft_transition/manifest.json').read_text())
    permission=request(f'https://api.github.com/repos/{args.archive_repo}',token=token)
    assert permission['permissions']['push'],'No archive write permission'
    print('archive authorization verified',flush=True)
    previous=None
    while True:
        try:
            state=request(args.control_url+'/state')
            assert args.training_revision.startswith(state['source_revision'])
            counts={status:sum(job['status']==status for job in state['jobs']) for status in ['pending','running','complete','failed']}
            summary=(state['phase'],tuple(counts.items()))
            if summary!=previous:
                print(datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),state['phase'],counts,flush=True)
                previous=summary
            if state['phase']=='complete':
                Path(args.token_file).unlink()
                print('ALL_EXPERIMENT_JOBS_COMPLETE; temporary archive credential removed',flush=True)
                break
            if state['phase'] not in ['failure_attention','generation']:
                jobs=extend_jobs(state,args.grid)
                if jobs:
                    archive_jobs(args,jobs,token,manifest)
                    with (root/'control/rate-extensions.jsonl').open('a') as handle:
                        handle.write(json.dumps({'source_revision':args.training_revision,'jobs':jobs})+'\n')
                    request(args.control_url+'/extend',{'jobs':jobs})
                    print('EXTENDED_GRID',[(job['temperature'],job['method'],job['learning_rate'],job['run_id']) for job in jobs],flush=True)
        except (urllib.error.URLError,ValueError,TimeoutError,subprocess.CalledProcessError) as error:
            print('WATCHER_ERROR',str(error),flush=True)
        time.sleep(15)


if __name__=='__main__':
    main(Args().parse_args())
