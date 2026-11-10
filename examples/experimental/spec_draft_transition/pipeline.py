"""CUDA-only multi-node experiment scheduler with independent GPU workers.

One coordinator owns job assignment and learning-rate selection. Each node
runs eight independent worker threads, each controlling a single-GPU process.
HTTP data servers expose only the data directory, never runtime credentials.
"""

import concurrent.futures
import functools
import http.server
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

from tap import Tap


class Args(Tap):
    mode: str
    root: str
    manifest: str
    coordinator: str
    node: int = 0
    slots: int = 8
    data_port: int = 19081
    control_port: int = 19082


def json_request(url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url,data=data,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=60) as response:
        return json.load(response)


class State:
    def __init__(self, args: Args) -> None:
        self.args = args
        self.root = Path(args.root)
        self.path = self.root/'control/state.json'
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.manifest = json.loads(Path(args.manifest).read_text())
        self.lock = threading.Lock()
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {
            'generation':{},'jobs':self.manifest['jobs'],'phase':'generation','selected_rates':{}}
        previous_revision=self.state.get('source_revision','42af7e7348')
        for job in self.state['jobs']:
            if job['status']!='pending':
                job.setdefault('source_revision',previous_revision)
        self.state.setdefault('claims_held',False)
        self.state['source_revision']=(self.root/'repo/SOURCE_REVISION').read_text().strip() if (self.root/'repo/SOURCE_REVISION').exists() else 'unrecorded'
        self.save()

    def save(self) -> None:
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.state,indent=2))
        temporary.replace(self.path)

    def finish_tuning(self) -> None:
        if any(job['status']=='failed' for job in self.state['jobs']):
            self.state['phase']='failure_attention'
            return
        tuning = [job for job in self.state['jobs'] if job['stage']=='tuning']
        for temperature in self.manifest['temperatures']:
            for method in self.manifest['methods']:
                group = [job for job in tuning if job['temperature']==temperature and job['method']==method]
                if not all(job['status']=='complete' for job in group):
                    continue
                candidates = {}
                for job in tuning:
                    if job['temperature']==temperature and job['method']==method:
                        candidates.setdefault(job['learning_rate'],[]).append(job['result']['validation']['expected_prefix'])
                scores = {rate:sum(values)/len(values) for rate,values in candidates.items()}
                best = max(sorted(scores),key=scores.get)
                key = f'{temperature}/{method}'
                self.state['selected_rates'][key] = {'rate':best,'scores':scores,'boundary':best in [min(scores),max(scores)]}
                for job in self.state['jobs']:
                    if job['stage']=='confirmation' and job['status']=='pending' and job['temperature']==temperature and job['method']==method and best not in [min(scores),max(scores)]:
                        job['learning_rate'] = best
        # A boundary winner requires extending the predeclared rate grid before
        # confirmation, rather than treating an unbracketed search as tuned.
        if not all(job['status']=='complete' for job in tuning):
            self.state['phase']='tuning'
        elif any(item['boundary'] for item in self.state['selected_rates'].values()):
            self.state['phase']='rate_boundary_attention'
        else:
            self.state['phase']='confirmation'

    def action(self, path: str, payload: dict) -> dict:
        with self.lock:
            if path=='/generation':
                self.state['generation'][str(payload['slot'])]=payload
                if len(self.state['generation'])==self.manifest['shards']:
                    self.state['phase']='tuning'
                self.save()
                return {'ok':True}
            if path=='/claim':
                phase=self.state['phase']
                if self.state['claims_held']:
                    return {'phase':phase,'job':None}
                if phase in ['tuning','confirmation','rate_boundary_attention']:
                    for job in sorted(self.state['jobs'], key=lambda item:item['stage']!='tuning'):
                        ready = job['stage']=='tuning' or (job['stage']=='confirmation' and job['learning_rate'] is not None)
                        if ready and job['status']=='pending':
                            job.update(status='running',slot=payload['slot'],started=time.time())
                            job['source_revision']=self.state['source_revision']
                            self.save()
                            return {'job':job}
                return {'phase':phase,'job':None}
            if path=='/complete':
                job=next(job for job in self.state['jobs'] if job['run_id']==payload['run_id'])
                job.update(status='complete' if payload['success'] else 'failed',result=payload,finished=time.time())
                if not payload['success']:
                    self.state['phase']='failure_attention'
                elif job['stage']=='tuning':
                    self.finish_tuning()
                elif all(item['status']=='complete' for item in self.state['jobs']):
                    self.state['phase']='complete'
                self.save()
                return {'ok':True}
            if path=='/extend':
                assert self.state['phase'] in ['rate_boundary_attention','tuning']
                existing = {job['run_id'] for job in self.state['jobs']}
                for job in payload['jobs']:
                    assert job['run_id'] not in existing and job['stage']=='tuning' and job['status']=='pending'
                    existing.add(job['run_id'])
                    self.state['jobs'].append(job)
                self.state['phase']='tuning'
                self.save()
                return {'ok':True}
            if path=='/hold':
                self.state['claims_held']=bool(payload['held'])
                self.save()
                return {'ok':True,'claims_held':self.state['claims_held']}
            if path=='/source':
                assert self.state['claims_held']
                assert (self.root/'repo/SOURCE_REVISION').read_text().strip()==payload['revision']
                self.state['source_revision']=payload['revision']
                self.save()
                return {'ok':True}
            if path=='/state':
                return self.state
            raise ValueError(path)


def coordinator(args: Args) -> None:
    state=State(args)

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            result=state.action(urllib.parse.urlparse(self.path).path,{})
            body=json.dumps(result).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(body)

        def do_POST(self) -> None:
            payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            result=state.action(self.path,payload)
            self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(json.dumps(result).encode())

    server=http.server.ThreadingHTTPServer(('0.0.0.0',args.control_port),Handler)
    print('coordinator ready',flush=True)
    server.serve_forever()


def sync_data(args: Args, manifest: dict, lock: threading.Lock) -> None:
    root=Path(args.root)/'data'
    with lock:
        marker=root/'all-shards-ready.json'
        if marker.exists():
            return
        for shard in range(manifest['shards']):
            node=shard//args.slots
            if node==args.node:
                continue
            for temperature in manifest['temperatures']:
                name=f't{temperature}-shard{shard:02d}.jsonl'
                for filename in [name,name.replace('.jsonl','.done')]:
                    url=f"http://{manifest['nodes'][node]}:{args.data_port}/generated/{filename}"
                    destination=root/'generated'/filename
                    temporary=destination.with_suffix(destination.suffix+'.download')
                    with urllib.request.urlopen(url,timeout=300) as response, temporary.open('wb') as handle:
                        shutil.copyfileobj(response,handle)
                    temporary.replace(destination)
        marker.write_text(json.dumps({'shards':manifest['shards'],'temperatures':manifest['temperatures']}))


def execute_job(args: Args, job: dict, gpu: int, environment: dict) -> dict:
    root=Path(args.root)
    output=root/'runs'/job['run_id']
    command=[sys.executable,'-m','examples.experimental.spec_draft_transition.train',
        '--teacher',str(root/'models/Qwen3-4B'),'--student',str(root/'models/Qwen3-0.6B'),
        '--data',str(root/'data/generated'),'--output',str(output),'--run_id',job['run_id'],
        '--temperature',str(job['temperature']),'--method',job['method'],'--learning_rate',str(job['learning_rate']),
        '--seed',str(job['seed']),'--updates',str(job['updates'])]
    log_path=root/'logs'/f"{job['run_id']}.log"
    with log_path.open('w') as log:
        code=subprocess.run(command,env=environment,stdout=log,stderr=subprocess.STDOUT).returncode
        if code:
            return {'success':False,'returncode':code,'log':str(log_path)}
        checkpoint=output/f"checkpoint-{job['updates']:04d}"
        validation=output/'validation-prefix.json'
        evaluation=[sys.executable,'-m','examples.experimental.spec_draft_transition.evaluate',
            '--teacher',str(root/'models/Qwen3-4B'),'--student',str(checkpoint),
            '--data',str(root/'data/generated'),'--output',str(validation),
            '--temperature',str(job['temperature']),'--split','validation','--prompts','128',
            '--repetitions','4','--expected']
        code=subprocess.run(evaluation,env=environment,stdout=log,stderr=subprocess.STDOUT).returncode
        if code:
            return {'success':False,'returncode':code,'log':str(log_path)}
        if job['stage']=='confirmation':
            for draft in [8,4,16]:
                test=output/f'test-k{draft}.jsonl'
                prompts=2000 if draft==8 else 256
                evaluation=[sys.executable,'-m','examples.experimental.spec_draft_transition.evaluate',
                    '--teacher',str(root/'models/Qwen3-4B'),'--student',str(checkpoint),
                    '--data',str(root/'data/generated'),'--output',str(test),
                    '--temperature',str(job['temperature']),'--prompts',str(prompts),'--draft',str(draft)]
                code=subprocess.run(evaluation,env=environment,stdout=log,stderr=subprocess.STDOUT).returncode
                if code:
                    return {'success':False,'returncode':code,'log':str(log_path)}
    return {'success':True,'validation':json.loads(validation.read_text()),'node':args.node,'gpu':gpu,
            'output':str(output),'wandb_url':(output/'wandb-url.txt').read_text()}


def worker(args: Args, gpu: int, manifest: dict, sync_lock: threading.Lock) -> None:
    root=Path(args.root)
    slot=args.node*args.slots+gpu
    while slot == 0 and manifest.get('benchmark_run_id') and not (root/'benchmark-gate.json').exists():
        time.sleep(5)
    environment={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'WANDB_API_KEY_FILE':str(root/'wandb-key'),
                 'HF_HOME':str(root/'hf-cache'),'HF_HUB_CACHE':str(root/'hf-cache/hub'),
                 'HF_DATASETS_CACHE':str(root/'hf-cache/datasets'),'OMP_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false'}
    command=[sys.executable,'-m','examples.experimental.spec_draft_transition.data','--mode','generate',
             '--root',str(root/'data'),'--teacher',str(root/'models/Qwen3-4B'),'--shard',str(slot),'--shards',str(manifest['shards'])]
    with (root/'logs'/f'generation-{slot:02d}.log').open('w') as log:
        result=subprocess.run(command,env=environment,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Generation failed slot={slot} code={result.returncode}')
    json_request(args.coordinator+'/generation',{'slot':slot,'node':args.node})
    while json_request(args.coordinator+'/state')['phase']=='generation':
        time.sleep(5)
    sync_data(args,manifest,sync_lock)
    while True:
        assignment=json_request(args.coordinator+'/claim',{'slot':slot})
        if assignment.get('phase')=='complete':
            return
        job=assignment.get('job')
        if job is None:
            time.sleep(5)
            continue
        result=execute_job(args,job,gpu,environment)
        json_request(args.coordinator+'/complete',{'run_id':job['run_id'],**result})
        print(json.dumps({'slot':slot,'run_id':job['run_id'],**result}),flush=True)


def node(args: Args) -> None:
    root=Path(args.root)
    os.chdir(root/'repo')
    (root/'logs').mkdir(exist_ok=True)
    (root/'runs').mkdir(exist_ok=True)
    (root/'data/generated').mkdir(exist_ok=True)
    manifest=json.loads(Path(args.manifest).read_text())
    handler=functools.partial(http.server.SimpleHTTPRequestHandler,directory=str(root/'data'))
    server=http.server.ThreadingHTTPServer(('0.0.0.0',args.data_port),handler)
    # Postprocessing fetches metrics after all workers finish; keep HTTP alive.
    threading.Thread(target=server.serve_forever,daemon=False).start()
    lock=threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.slots) as pool:
        futures=[pool.submit(worker,args,gpu,manifest,lock) for gpu in range(args.slots)]
        for future in concurrent.futures.as_completed(futures):
            future.result()


if __name__=='__main__':
    config=Args().parse_args()
    if config.mode=='coordinator':
        coordinator(config)
    elif config.mode=='node':
        node(config)
    else:
        raise ValueError(config.mode)
