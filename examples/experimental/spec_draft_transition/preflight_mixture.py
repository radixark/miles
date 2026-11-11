"""Check the constant blend in real model training before releasing full fits."""

import importlib.metadata
import json
import math
import os
import subprocess
import time
from pathlib import Path

from tap import Tap


class Args(Tap):
    root: str


def main(args: Args) -> None:
    root=Path(args.root)
    manifest=json.loads((root/'manifest.json').read_text())
    baseline=Path(manifest['baseline_root'])
    while not (baseline/'report/reference-throughput.json').exists():
        time.sleep(15)
    source=(root/'repo/SOURCE_REVISION').read_text().strip()
    environment={**os.environ,'CUDA_VISIBLE_DEVICES':'0','PYTHONPATH':str(root/'repo'),
        'WANDB_BASE_URL':'https://api.forge.coreweave.com','WANDB_API_KEY_FILE':str(root/'wandb-key'),
        'HF_HOME':str(root/'hf-cache'),'HF_HUB_CACHE':str(root/'hf-cache/hub'),
        'HF_DATASETS_CACHE':str(root/'hf-cache/datasets'),'OMP_NUM_THREADS':'4',
        'OPENBLAS_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false'}
    python=str(root/'venv/bin/python')
    helper=['python3','/root/.cache/ds_helper/ds_helper.py']
    subprocess.run([*helper,'pause'],check=True)
    try:
        with (root/'logs/amp-probe.log').open('w') as log:
            subprocess.run([python,'-m','examples.experimental.spec_draft_transition.probe_amp',
                '--teacher',str(root/'models/Qwen3-4B'),'--student',str(root/'models/Qwen3-0.6B'),
                '--output',str(root/'report/amp-probe.json')],cwd=root/'repo',env=environment,
                stdout=log,stderr=subprocess.STDOUT,check=True)
        cases=json.loads((root/'report/amp-probe.json').read_text())
        assert len(cases)==28 and any(case['ce_weight']==0.1 for case in cases)
        results=[]
        for label,run_id,updates,tokens,seed in [
            ('smoke',manifest['smoke_run_id'],3,128,51001),
            ('benchmark',manifest['benchmark_run_id'],2,16384,51002)]:
            output=root/'runs'/run_id
            command=[python,'-m','examples.experimental.spec_draft_transition.train',
                '--teacher',str(root/'models/Qwen3-4B'),'--student',str(root/'models/Qwen3-0.6B'),
                '--data',str(root/'data/generated'),'--output',str(output),'--run_id',run_id,
                '--temperature','1.0','--method','mix90','--learning_rate','3e-5','--seed',str(seed),
                '--updates',str(updates),'--tokens_per_update',str(tokens),'--warmup','1',
                '--validation_interval',str(updates),'--validation_tokens','128']
            with (root/'logs'/f'{run_id}.log').open('w') as log:
                subprocess.run(command,cwd=root/'repo',env=environment,stdout=log,
                               stderr=subprocess.STDOUT,check=True)
            records=[json.loads(line) for line in (output/'metrics.jsonl').read_text().splitlines()]
            training=[row for row in records if row['kind']=='training']
            assert len(training)==updates
            assert all(row['ce_weight']==0.1 for row in training)
            assert all(math.isfinite(row[key]) for row in training for key in ['loss','gradient_norm'])
            config=json.loads((output/'config.json').read_text())
            assert config['source_revision']==source and config['method']=='mix90'
            result={'label':label,'run_id':run_id,'updates':updates,'tokens_per_update':tokens,
                    'training_records':training,'wandb_url':(output/'wandb-url.txt').read_text().strip()}
            results.append(result)
            print(json.dumps(result),flush=True)
        provenance={'source_revision':source,'checks':results,'amp_cases':len(cases),
            'versions':{name:importlib.metadata.version(name) for name in
                        ['torch','transformers','sglang','polars','wandb']},
            'nonsecret_environment':{name:environment[name] for name in
                ['CUDA_VISIBLE_DEVICES','PYTHONPATH','WANDB_BASE_URL','HF_HOME','HF_HUB_CACHE',
                 'HF_DATASETS_CACHE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','TOKENIZERS_PARALLELISM']}}
        (root/'report/preflight-ready.json').write_text(json.dumps(provenance,indent=2))
        (root/'benchmark-gate.json').write_text(json.dumps(provenance,indent=2))
        print('FIXED_MIXTURE_PREFLIGHT_PASSED',flush=True)
    finally:
        subprocess.run([*helper,'resume'],check=True)


if __name__=='__main__':
    main(Args().parse_args())
