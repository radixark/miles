"""Audit the completed adaptive cohort and compare it with frozen baselines."""

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from tap import Tap

from examples.experimental.spec_draft_transition.analyze import bootstrap, read_shards

matplotlib.use('Agg')

METHODS = ['ce', 'l1', 'smooth', 'mix90', 'adaptive']
LABELS = ['CE', 'L1', 'Smooth CE→L1', 'Fixed mixture', 'L1-triggered transition']
COLORS = ['#2563eb', '#dc2626', '#059669', '#9333ea', '#d97706']


class Args(Tap):
    root: str
    control_url: str = 'http://127.0.0.1:19088'
    resamples: int = 20000
    seed: int = 20261108
    remote_data_url: str = ''


def collect_remote(root: Path, state: dict, url: str, control_url: str) -> None:
    """Copy compact completed-run records, leaving checkpoints on their node."""
    names=['config.json','complete.json','metrics.jsonl','transition.json',
           'wandb-url.txt','validation-prefix.json','test-k8.jsonl','test-k4.jsonl','test-k16.jsonl']
    for job in state['jobs']:
        if job['stage']!='confirmation' or job['slot']//8==0:
            continue
        assert job['slot']//8==1 and job['status']=='complete'
        directory=root/'runs'/job['run_id']
        directory.mkdir(exist_ok=True)
        for name in names:
            destination=directory/name
            temporary=destination.with_suffix(destination.suffix+'.download')
            with urllib.request.urlopen(url.rstrip('/')+'/results/'+job['run_id']+'/'+name,timeout=60) as response:
                temporary.write_bytes(response.read())
            temporary.replace(destination)
        config=json.loads((directory/'config.json').read_text())
        assert config['run_id']==job['run_id'] and config['seed']==job['seed']
        assert config['temperature']==job['temperature'] and config['stage']=='confirmation'
        assert config['source_revision']==state['source_revision']
    subprocess.run([sys.executable,'-m','examples.experimental.spec_draft_transition.collect',
                    '--root',str(root),'--control_url',control_url,'--node','1'],check=True)


def audit_switch(directory: Path) -> dict:
    records = [json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
    checks = [r for r in records if r['kind']=='transition_check']
    assert [r['step'] for r in checks] == list(range(0,2001,100))
    eligible = []
    for index in range(5,20):
        earlier = min(r['validation_l1'] for r in checks[:index-2])
        best = min(r['validation_l1'] for r in checks[:index+1])
        if (earlier-best)/earlier < .005:
            eligible.append(checks[index]['step'])
    expected = min(eligible) if eligible else None
    triggers = [r['step'] for r in checks if r['triggered']]
    assert triggers == ([] if expected is None else [expected])
    training = [r for r in records if r['kind']=='training']
    assert len(training)==2000
    for row in training:
        weights = (row['hard_ce_weight'],row['ce_weight'],row['l1_weight'])
        assert weights == ((0.,.1,.9) if expected is not None and row['update']>expected else (.1,.9,0.))
    return dict(switch_step=expected,final_l1=checks[-1]['validation_l1'])


def main(args: Args) -> None:
    root = Path(args.root)
    previous = None
    while True:
        with urllib.request.urlopen(args.control_url+'/state',timeout=30) as response:
            state=json.load(response)
        counts={s:sum(j['status']==s for j in state['jobs']) for s in ['pending','running','complete','failed']}
        if counts != previous:
            print(json.dumps({'phase':state['phase'],**counts}),flush=True)
            previous=counts
        if counts['failed']:
            raise RuntimeError('Failed fits require investigation before analysis')
        if state['phase']=='complete':
            break
        time.sleep(15)
    subprocess.run([sys.executable,'-m','examples.experimental.spec_draft_transition.collect',
                    '--root',str(root),'--control_url',args.control_url,'--node','0'],check=True)
    if args.remote_data_url:
        collect_remote(root,state,args.remote_data_url,args.control_url)
    report=root/'report'
    sources=[root/'baseline/original',root/'baseline/mixture',report]
    samples=pl.concat([read_shards(source,'samples') for source in sources],how='diagonal_relaxed')
    curves=pl.concat([read_shards(source,'curves') for source in sources],how='diagonal_relaxed')
    jobs=pl.concat([read_shards(source,'jobs') for source in sources],how='diagonal_relaxed')
    assert jobs['run_id'].n_unique()==150 and jobs.height==150
    samples.write_parquet(report/'all-samples.parquet')
    curves.write_parquet(report/'all-curves.parquet')
    jobs.write_parquet(report/'all-jobs.parquet')
    seeds=list(range(20001,20011))
    temperatures=[.6,1.,1.4]
    accepted=np.empty((3,5,10,2000))
    blocks=np.empty_like(accepted)
    identities=None
    for i,temp in enumerate(temperatures):
        for j,method in enumerate(METHODS):
            for k,seed in enumerate(seeds):
                part=samples.filter((pl.col('draft')==8)&(pl.col('temperature')==temp)&(pl.col('method')==method)&(pl.col('seed')==seed)).sort('prompt_id')
                assert part.height==2000,(temp,method,seed,part.height)
                if identities is None:
                    identities=part['prompt_id'].to_list()
                assert part['prompt_id'].to_list()==identities
                accepted[i,j,k]=part['accepted'].to_numpy()
                blocks[i,j,k]=part['blocks'].to_numpy()
    observed=(accepted.sum(-1)/blocks.sum(-1)).mean(-1)
    draws=bootstrap(accepted,blocks,args.resamples,np.random.default_rng(args.seed))
    results=[]
    contrasts=[]
    for i,temp in enumerate(temperatures):
        for j,method in enumerate(METHODS):
            lo,hi=np.quantile(draws[i,j],[.025,.975])
            results.append(dict(temperature=temp,method=method,mean=float(observed[i,j]),
                standard_error=float(draws[i,j].std(ddof=1)),lower95=float(lo),upper95=float(hi)))
        for j,baseline in enumerate(METHODS[:-1]):
            lo,hi=np.quantile(draws[i,4]-draws[i,j],[.05/24,1-.05/24])
            contrasts.append(dict(temperature=temp,baseline=baseline,difference=float(observed[i,4]-observed[i,j]),
                                  lower_adjusted=float(lo),upper_adjusted=float(hi),family_size=12))
    switches=[]
    for job in state['jobs']:
        if job['stage']=='confirmation':
            switches.append({**{key:job[key] for key in ['run_id','temperature','seed','learning_rate']},
                             **audit_switch(Path(job['result']['output']))})
    summary=dict(experiment_id='261110-a2d703fc',models=150,means=results,
                 adaptive_contrasts=contrasts,switches=switches,selected_rates=state['selected_rates'],
                 resamples=args.resamples,bootstrap='paired crossed training-seed/prompt',
                 error_bars='one bootstrap standard error',contrast_family='12 Bonferroni-adjusted 95% intervals')
    (report/'adaptive-summary.json').write_text(json.dumps(summary,indent=2))
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
    for i,temp in enumerate(temperatures):
        axis=axes[i]
        axis.bar(np.arange(5),observed[i],yerr=draws[i].std(axis=-1,ddof=1),color=COLORS,capsize=4,width=.65)
        axis.set_xticks(np.arange(5),LABELS,rotation=30,ha='right',fontsize=9)
        axis.set_title(f'Teacher temperature {temp:g}')
        axis.set_ylim(2.4,3.0)
        axis.grid(axis='y',alpha=.2)
        axis.set_axisbelow(True)
    axes[0].set_ylabel('Accepted draft tokens per round (K=8)')
    fig.suptitle('Mean ± one bootstrap standard error · 10 seeds × 2,000 prompts')
    fig.tight_layout()
    for suffix in ['svg','png']:
        fig.savefig(report/f'accepted-prefix.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,3.5),sharey=True)
    steps=np.arange(0,2001,200)
    for i,temp in enumerate(temperatures):
        for method,label,color in zip(METHODS,LABELS,COLORS):
            group=curves.filter((pl.col('temperature')==temp)&(pl.col('method')==method)&(pl.col('update')%200==0))
            values=np.asarray([group.filter(pl.col('seed')==seed).sort('update')['l1'].to_numpy() for seed in seeds])
            assert values.shape==(10,11),(temp,method,values.shape)
            mean=values.mean(0)
            se=values.std(0,ddof=1)/np.sqrt(10)
            axes[i].plot(steps,mean,color=color,label=label)
            axes[i].fill_between(steps,mean-se,mean+se,color=color,alpha=.13)
        axes[i].set_title(f'Teacher temperature {temp:g}')
        axes[i].set_xlabel('Training updates')
        axes[i].grid(alpha=.2)
    axes[0].set_ylabel('Held-out full-distribution L1')
    axes[0].legend(fontsize=8)
    fig.suptitle('Training curves · mean ± one standard error across 10 seeds')
    fig.tight_layout()
    fig.savefig(report/'training-curves.svg',bbox_inches='tight')
    fig.savefig(report/'training-curves.png',dpi=180,bbox_inches='tight')
    plt.close(fig)
    lines=[]
    for temp in temperatures:
        group=[r for r in switches if r['temperature']==temp]
        triggered=[r['switch_step'] for r in group if r['switch_step'] is not None]
        text=f'T={temp:g}: '+(f'switch step median {np.median(triggered):g}, range {min(triggered)}–{max(triggered)}' if triggered else 'no switches')
        lines.append('- '+text+f'; {10-len(triggered)}/10 runs never switched.')
    text='''# Drafter distillation: validation-triggered loss transition

Frozen Qwen3-4B teacher → pretrained Qwen3-0.6B drafter. Teacher temperatures
0.6/1.0/1.4; drafter temperature 1. Same teacher-generated UltraChat continuations
and 2,000 held-out test prompts as the four existing baselines; no entropy filtering.
Each method uses ten paired confirmation seeds per temperature. Each fit trains
for 2,000 updates × 16,384 response positions, with independently validation-tuned
learning rates. New fits run across two eight-H200 boxes, shi-h200-2 and shi-h200-4.

## New schedule

Start with `0.1 hard CE + 0.9 soft CE`. Hard CE uses the exact sampled teacher
next-token ID saved in the training continuation; soft CE uses the complete teacher
distribution. Every 100 updates, measure L1 on fixed held-out validation positions
(4,096 positions, up to 32 per conversation). At the first check from update 500
through 1,900 where the best validation L1 improved by less than 0.5% over the
preceding 300 updates, switch to `0.9 L1 + 0.1 soft CE`. Preserve optimizer state
and the global learning-rate schedule. Never force a switch. L1 sums over the
151,936-token vocabulary and averages over positions; coefficients are not rescaled.

For reference, smooth CE→L1 uses `w(u) CE + (1−w(u)) L1`, with
`w(u) = (1 + cos(pi*u/1999))/2` for updates `u=0,…,1999`.
The fixed mixture uses `0.9 L1 + 0.1 soft CE` throughout.

## Results

![Accepted draft tokens per round](accepted-prefix.svg)

Bars show mean ± one crossed seed/prompt bootstrap standard error (20,000 draws).
The y-axis starts at 2.4. Evaluation uses actual sequential speculative rejection
sampling with eight proposals, at most 32 rounds and EOS stopping. Accepted tokens,
residual corrections and bonus tokens update history. The metric counts accepted
draft proposals only; it is not a production serving speedup.

![Held-out L1 during training](training-curves.svg)

Training curves show mean ± one standard error across seeds. Actual transition
decisions (validation data only):

'''+ '\n'.join(lines)+'''

This comparison changes both the initial hard/soft CE mixture and the transition
rule, so it cannot isolate the benefit of the trigger alone.
'''
    (report/'gist-results.md').write_text(text)
    (report/'ANALYSIS_COMPLETE').write_text(json.dumps({'models':150,'adaptive_models':30,'all_switches_audited':True}))
    print('ADAPTIVE_COMPARISON_COMPLETE',flush=True)


if __name__=='__main__':
    main(Args().parse_args())
