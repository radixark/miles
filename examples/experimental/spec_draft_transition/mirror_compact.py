"""Mirror compact second-node records for interim analyses; never copy weights."""

import http.client
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
    remote_data_url: str
    source_revision: str = '36937b517159a618e6c4cdbf66730af5b408054a'


def download(url: str) -> bytes | None:
    try:
        with urllib.request.urlopen(url,timeout=15) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        if error.code==404:
            return None
        raise


def mirror(root: Path, state: dict, args: Args) -> dict:
    assert state['source_revision']==args.source_revision
    jobs=[j for j in state['jobs'] if j['stage']=='confirmation' and j.get('slot',0)//8==1]
    saved=0
    for job in jobs:
        base=args.remote_data_url.rstrip('/')+'/results/'+job['run_id']+'/'
        raw=download(base+'config.json')
        if raw is None:
            continue
        config=json.loads(raw)
        assert config['run_id']==job['run_id'] and config['seed']==job['seed']
        assert config['source_revision']==args.source_revision and config['temperature']==job['temperature']
        assert config['stage']=='confirmation'
        directory=root/'runs'/job['run_id']
        directory.mkdir(exist_ok=True)
        for name in ['config.json','wandb-url.txt','complete.json','validation-prefix.json',
                     'transition.json','metrics.jsonl','test-k8.jsonl','test-k4.jsonl','test-k16.jsonl']:
            destination=directory/name
            if name in ['wandb-url.txt','complete.json','validation-prefix.json'] and destination.exists():
                continue
            data=raw if name=='config.json' else download(base+name)
            if data is None:
                continue
            if name.endswith('.jsonl'):
                # Readers see only complete records while the remote writer appends.
                data=data[:data.rfind(b'\n')+1]
                for line in data.splitlines():
                    json.loads(line)
            elif name.endswith('.json'):
                json.loads(data)
            temporary=destination.with_suffix(destination.suffix+'.mirror')
            temporary.write_bytes(data)
            temporary.replace(destination)
            saved+=1
    return dict(remote_jobs=len(jobs),saved_files=saved,phase=state['phase'],
                checked_at_pt=datetime.now(ZoneInfo('America/Los_Angeles')).isoformat())


def main(args: Args) -> None:
    root=Path(args.root)
    while True:
        state=json.loads((root/'control/state.json').read_text())
        try:
            status=mirror(root,state,args)
            (root/'report/remote-mirror-status.json').write_text(json.dumps(status,indent=2))
            print(json.dumps(status),flush=True)
        except (OSError,http.client.IncompleteRead,json.JSONDecodeError) as error:
            print('MIRROR_RETRY',type(error).__name__,flush=True)
        if state['phase']=='complete':
            break
        for _ in range(4):
            time.sleep(15)


if __name__=='__main__':
    main(Args().parse_args())
