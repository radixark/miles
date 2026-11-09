"""Run paired reference timing replicates on free GPUs after scientific fits finish."""

import concurrent.futures
import json
import os
import queue
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from tap import Tap


class Args(Tap):
    root: str
    node: int
    control_url: str
    manifest: str
    slots: int = 8
    data_port: int = 19081


def read_json(url: str) -> dict | list:
    with urllib.request.urlopen(url,timeout=60) as response:
        return json.load(response)


def run_task(root: Path, job: dict, repetition: int, gpu: int) -> dict:
    output=root/'report'/f'benchmark-{job["run_id"]}-rep{repetition}.json'
    command=[sys.executable,'-m','examples.experimental.spec_draft_transition.benchmark',
             '--teacher',str(root/'models/Qwen3-4B'),
             '--student',str(Path(job['result']['output'])/'checkpoint-2000'),
             '--data',str(root/'data/generated'),'--output',str(output),
             '--temperature',str(job['temperature']),'--prompts','128','--max_tokens','256',
             '--repetitions','1','--repetition_start',str(repetition),'--seed','22931']
    environment={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false'}
    with output.with_suffix('.log').open('w') as log:
        subprocess.run(command,env=environment,cwd=root/'analysis-repo',stdout=log,stderr=subprocess.STDOUT,check=True)
    result=json.loads(output.read_text())
    result.update({key:job[key] for key in ['run_id','temperature','method','seed','learning_rate']})
    result.update(host=socket.gethostname(),gpu=gpu,source_revision=(root/'analysis-repo/SOURCE_REVISION').read_text().strip(),
                  wandb_url=job['result']['wandb_url'].strip())
    print('BENCHMARK_COMPLETE',job['run_id'],repetition,gpu,flush=True)
    return result


def main(args: Args) -> None:
    root=Path(args.root)
    while True:
        state=read_json(args.control_url+'/state')
        if state['phase']=='complete':
            break
        time.sleep(15)
    jobs=[job for job in state['jobs'] if job['stage']=='confirmation' and job['seed']==20001]
    assert len(jobs)==9 and all(job['status']=='complete' for job in jobs)
    tasks=queue.Queue()
    for job in jobs:
        if job['slot']//8==args.node:
            for repetition in range(3):
                tasks.put((job,repetition))
    (root/'report').mkdir(exist_ok=True)

    def worker(gpu: int) -> list[dict]:
        results=[]
        while True:
            try:
                job,repetition=tasks.get_nowait()
            except queue.Empty:
                return results
            results.append(run_task(root,job,repetition,gpu))

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.slots) as pool:
        results=[result for group in pool.map(worker,range(args.slots)) for result in group]
    path=root/'report'/f'benchmarks-node{args.node}.json'
    path.write_text(json.dumps(results,indent=2))
    transfer=root/'data/report';transfer.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(path,transfer/path.name)
    if args.node!=0:
        return
    manifest=json.loads(Path(args.manifest).read_text())
    for node,host in enumerate(manifest['nodes']):
        if node==0:
            continue
        url=f'http://{host}:{args.data_port}/report/benchmarks-node{node}.json'
        while True:
            try:
                part=read_json(url)
            except (urllib.error.URLError,TimeoutError):
                time.sleep(15)
                continue
            results.extend(part)
            break
    assert len(results)==27
    (root/'report/reference-throughput.json').write_text(json.dumps(results,indent=2))
    print('ALL_REFERENCE_BENCHMARKS_COMPLETE',flush=True)


if __name__=='__main__':
    main(Args().parse_args())
