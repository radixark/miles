"""Round-trip a real completed run through HTTP transfer and the real collector."""

import copy
import io
import json
import shutil
import subprocess
import tempfile
import threading
import urllib.request
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import polars as pl
from tap import Tap

from examples.experimental.spec_draft_transition.analyze_adaptive import audit_switch, collect_remote
from examples.experimental.spec_draft_transition.collect import collect


class Args(Tap):
    root: str


def main(args: Args) -> None:
    root=Path(args.root)
    state=json.loads((root/'control/state.json').read_text())
    job=next(j for j in state['jobs'] if j['stage']=='confirmation' and j['status']=='complete' and j['slot']//8==0)
    with tempfile.TemporaryDirectory(prefix='collector-preflight-',dir=root/'report') as temporary:
        fixture=Path(temporary)
        destination=fixture/'destination'
        (destination/'runs').mkdir(parents=True)
        export=fixture/'http/results'/job['run_id']
        export.mkdir(parents=True)
        for name in ['config.json','complete.json','metrics.jsonl','transition.json','wandb-url.txt',
                     'validation-prefix.json','test-k8.jsonl','test-k4.jsonl','test-k16.jsonl']:
            shutil.copy2(Path(job['result']['output'])/name,export/name)
        remote=copy.deepcopy(job)
        remote['slot']=8
        remote['result']['output']=str(destination/'runs'/job['run_id'])
        completed={'phase':'complete','source_revision':state['source_revision'],'jobs':[remote]}
        handler=partial(SimpleHTTPRequestHandler,directory=str(fixture/'http'))
        with ThreadingHTTPServer(('127.0.0.1',0),handler) as server:
            thread=threading.Thread(target=server.serve_forever,daemon=True)
            thread.start()
            with patch.object(subprocess,'run') as launch:
                collect_remote(destination,completed,f'http://127.0.0.1:{server.server_port}','http://fixture.invalid')
                command=launch.call_args.args[0]
                assert command[-2:]==['--node','1']
            server.shutdown()
            thread.join()
        with patch.object(urllib.request,'urlopen',return_value=io.BytesIO(json.dumps(completed).encode())):
            collect(SimpleNamespace(root=str(destination),control_url='http://fixture.invalid',node=1))
        assert pl.read_parquet(destination/'report/jobs-node1.parquet').height==1
        assert pl.read_parquet(destination/'report/samples-node1.parquet').height==2512
        assert pl.read_parquet(destination/'report/curves-node1.parquet').height==21
        decision=audit_switch(destination/'runs'/job['run_id'])
    result={'passed':True,'source_run':job['run_id'],'samples':2512,'curves':21,'transition_audit':decision}
    (root/'report/multinode-preflight.json').write_text(json.dumps(result,indent=2))
    print('REAL_HTTP_COLLECTOR_PREFLIGHT_PASSED',json.dumps(result),flush=True)


if __name__=='__main__':
    main(Args().parse_args())
