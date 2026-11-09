"""Paired seed/prompt bootstrap and common-metric training curves."""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from tap import Tap

matplotlib.use('Agg')


class Args(Tap):
    root: str
    resamples: int = 20000
    seed: int = 20261108


def arrays(samples: pl.DataFrame, expected_prompts: int = 2000) -> tuple[np.ndarray, np.ndarray, list[float], list[int], list[str]]:
    temperatures=sorted(samples['temperature'].unique().to_list())
    seeds=sorted(samples['seed'].unique().to_list())
    methods=['ce','l1','smooth']
    accepted=np.empty((len(temperatures),3,len(seeds),expected_prompts))
    blocks=np.empty_like(accepted)
    identities=None
    for i,temperature in enumerate(temperatures):
        for j,method in enumerate(methods):
            for k,seed in enumerate(seeds):
                part=samples.filter((pl.col('temperature')==temperature)&(pl.col('method')==method)&(pl.col('seed')==seed)).sort('prompt_id')
                assert len(part)==expected_prompts
                if identities is None:
                    identities=part['prompt_id'].to_list()
                assert part['prompt_id'].to_list()==identities
                accepted[i,j,k]=part['accepted'].to_numpy()
                blocks[i,j,k]=part['blocks'].to_numpy()
    assert len(temperatures)==3 and len(seeds)==10
    return accepted,blocks,temperatures,seeds,methods


def bootstrap(accepted: np.ndarray, blocks: np.ndarray, resamples: int, rng: np.random.Generator) -> np.ndarray:
    output=np.empty((*accepted.shape[:2],resamples))
    seeds,prompts=accepted.shape[-2:]
    for start in range(0,resamples,100):
        count=min(100,resamples-start)
        prompt_counts=rng.multinomial(prompts,np.full(prompts,1/prompts),size=count)
        seed_counts=rng.multinomial(seeds,np.full(seeds,1/seeds),size=count)
        a=(accepted.reshape(-1,prompts)@prompt_counts.T).reshape(*accepted.shape[:3],count)
        b=(blocks.reshape(-1,prompts)@prompt_counts.T).reshape(*blocks.shape[:3],count)
        output[...,start:start+count]=((a/b)*seed_counts.T).sum(axis=2)/seeds
    return output


def plot_curves(curves: pl.DataFrame, root: Path, temperatures: list[float], seeds: list[int],
                rng: np.random.Generator) -> None:
    colors={'ce':'#2563eb','l1':'#dc2626','smooth':'#059669'}
    labels={'ce':'CE','l1':'L1','smooth':'Smooth CE→L1'}
    figure,axes=plt.subplots(2,3,figsize=(13,7),sharex=True)
    differences,paired=plt.subplots(1,3,figsize=(13,3.5),sharex=True)
    weights=rng.multinomial(len(seeds),np.full(len(seeds),1/len(seeds)),size=20000)/len(seeds)
    for i,temperature in enumerate(temperatures):
        overlap={}
        for method in ['ce','l1','smooth']:
            group=curves.filter((pl.col('temperature')==temperature)&(pl.col('method')==method))
            steps=sorted(group['update'].unique().to_list())
            for row,metric in enumerate(['overlap','kl']):
                values=np.array([group.filter(pl.col('seed')==seed).sort('update')[metric].to_numpy() for seed in seeds])
                assert values.shape==(10,len(steps))
                if metric=='overlap':overlap[method]=values
                simulated=weights@values
                lo,hi=np.quantile(simulated,[.025,.975],axis=0)
                axes[row,i].plot(np.array(steps)*16384/1e6,values.mean(0),color=colors[method],label=labels[method])
                axes[row,i].fill_between(np.array(steps)*16384/1e6,lo,hi,color=colors[method],alpha=.15)
            axes[0,i].set_title(f'Teacher temperature {temperature}')
        for baseline in ['ce','l1']:
            values=overlap['smooth']-overlap[baseline]
            lo,hi=np.quantile(weights@values,[.025,.975],axis=0)
            paired[i].plot(np.array(steps)*16384/1e6,values.mean(0),color=colors[baseline],label='Smooth − '+labels[baseline])
            paired[i].fill_between(np.array(steps)*16384/1e6,lo,hi,color=colors[baseline],alpha=.15)
        paired[i].axhline(0,color='#666666',linestyle='--',linewidth=.8)
        paired[i].set_title(f'Teacher temperature {temperature}')
        paired[i].set_xlabel('Training response tokens (millions)')
        axes[1,i].set_xlabel('Training response tokens (millions)')
    axes[0,0].set_ylabel('Validation probability overlap')
    axes[1,0].set_ylabel('Validation KL (nats)')
    axes[0,0].legend(fontsize=8)
    paired[0].set_ylabel('Paired overlap difference')
    paired[0].legend(fontsize=8)
    figure.tight_layout();figure.savefig(root/'learning-curves.png',dpi=180);plt.close(figure)
    differences.tight_layout();differences.savefig(root/'paired-curves.png',dpi=180);plt.close(differences)


def main(args: Args) -> None:
    root=Path(args.root)
    samples=pl.read_parquet(root/'samples-node*.parquet')
    curves=pl.read_parquet(root/'curves-node*.parquet')
    jobs=pl.read_parquet(root/'jobs-node*.parquet')
    state=json.loads((root/'state.json').read_text())
    assert jobs['run_id'].n_unique()==90
    accepted,blocks,temperatures,seeds,methods=arrays(samples.filter(pl.col('draft')==8))
    observed=(accepted.sum(-1)/blocks.sum(-1)).mean(-1)
    rng=np.random.default_rng(args.seed)
    draws=bootstrap(accepted,blocks,args.resamples,rng)
    means,contrasts,interactions=[],[],[]
    for i,temperature in enumerate(temperatures):
        for j,method in enumerate(methods):
            lo,hi=np.quantile(draws[i,j],[.025,.975])
            means.append({'temperature':temperature,'method':method,'mean':float(observed[i,j]),'lower95':float(lo),'upper95':float(hi)})
        for j,baseline in enumerate(methods[:2]):
            delta=draws[i,2]-draws[i,j]
            lo,hi=np.quantile(delta,[.05/12,1-.05/12])
            contrasts.append({'temperature':temperature,'baseline':baseline,'difference':float(observed[i,2]-observed[i,j]),'lower_adjusted':float(lo),'upper_adjusted':float(hi)})
    for j,baseline in enumerate(methods[:2]):
        delta=(draws[2,2]-draws[2,j])-(draws[0,2]-draws[0,j])
        lo,hi=np.quantile(delta,[.025,.975])
        interactions.append({'baseline':baseline,'high_minus_low_gain':float((observed[2,2]-observed[2,j])-(observed[0,2]-observed[0,j])), 'lower95':float(lo),'upper95':float(hi)})
    ce_l1=[]
    for i,temperature in enumerate(temperatures):
        lo,hi=np.quantile(draws[i,0]-draws[i,1],[.025,.975])
        ce_l1.append({'temperature':temperature,'ce_minus_l1':float(observed[i,0]-observed[i,1]),
                      'lower95':float(lo),'upper95':float(hi)})
    delta=(draws[2,0]-draws[2,1])-(draws[0,0]-draws[0,1])
    lo,hi=np.quantile(delta,[.025,.975])
    ce_l1_interaction={'high_minus_low_ce_advantage':float((observed[2,0]-observed[2,1])-(observed[0,0]-observed[0,1])),
                       'lower95':float(lo),'upper95':float(hi)}
    secondary_decoding=[]
    for draft in sorted(samples['draft'].unique().to_list()):
        part=samples.filter(pl.col('draft')==draft)
        for metric in ['accepted','emitted']:
            if metric not in part.columns:
                continue
            a,b,_,_,_=arrays(part.with_columns(pl.col(metric).alias('accepted')),2000 if draft==8 else 256)
            estimates=(a.sum(-1)/b.sum(-1)).mean(-1)
            simulated=draws if draft==8 and metric=='accepted' else bootstrap(a,b,args.resamples,rng)
            for i,temperature in enumerate(temperatures):
                for j,method in enumerate(methods):
                    lo,hi=np.quantile(simulated[i,j],[.025,.975])
                    secondary_decoding.append({'draft':draft,'metric':metric,'temperature':temperature,'method':method,
                        'mean':float(estimates[i,j]),'lower95':float(lo),'upper95':float(hi)})
    summary={'means':means,'primary_contrasts':contrasts,'secondary_interactions':interactions,
             'secondary_ce_minus_l1':ce_l1,'secondary_ce_l1_interaction':ce_l1_interaction,
             'secondary_decoding':secondary_decoding,
             'interval':'Crossed bootstrap of training seeds and held-out prompts; six primary contrasts, Bonferroni nominal95% family coverage.',
             'resamples':args.resamples,'seeds':seeds,'evaluation_source':state['source_revision'],
             'training_sources':sorted({job.get('source_revision','unrecorded') for job in state.get('jobs',[])})}
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    pl.DataFrame(means).write_csv(root/'accepted-prefix-means.csv')
    pl.DataFrame(contrasts).write_csv(root/'paired-contrasts.csv')
    pl.DataFrame(secondary_decoding).write_csv(root/'secondary-decoding.csv')
    jobs.write_csv(root/'runs.csv')
    curves.write_parquet(root/'curves.parquet')
    samples.write_parquet(root/'samples.parquet')
    plot_curves(curves,root,temperatures,seeds,rng)
    figure,axes=plt.subplots(1,3,figsize=(10,3.5),sharey=True)
    for i,temperature in enumerate(temperatures):
        rows=[row for row in means if row['temperature']==temperature]
        axes[i].bar(range(3),[row['mean'] for row in rows],color=['#2563eb','#dc2626','#059669'])
        axes[i].errorbar(range(3),[row['mean'] for row in rows],yerr=[[row['mean']-row['lower95'] for row in rows],[row['upper95']-row['mean'] for row in rows]],fmt='none',color='black',capsize=3)
        axes[i].set_xticks(range(3),['CE','L1','Smooth'])
        axes[i].set_title(f'Temperature {temperature}')
    axes[0].set_ylabel('Accepted draft-prefix tokens per K8 round')
    figure.tight_layout();figure.savefig(root/'accepted-prefix.png',dpi=180);plt.close(figure)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main(Args().parse_args())
