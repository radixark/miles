"""Measure positional entropy on held-out teacher-generated histories."""

from pathlib import Path

import polars as pl
import torch
from tap import Tap
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.experimental.spec_draft_transition.train import batch, load_rows


class Args(Tap):
    teacher: str
    data: str
    output: str
    temperature: float
    prompts: int = 2000


@torch.no_grad()
def main(args: Args) -> None:
    torch.set_num_threads(4)
    device=torch.accelerator.current_accelerator()
    teacher=AutoModelForCausalLM.from_pretrained(args.teacher,dtype=torch.bfloat16,attn_implementation='sdpa').to(device).eval()
    tokenizer=AutoTokenizer.from_pretrained(args.teacher)
    rows=load_rows(args.data,args.temperature,'test')[:args.prompts]
    records=[]
    for start in range(0,len(rows),2):
        part=rows[start:start+2]
        ids,attention,_=batch(part,tokenizer.pad_token_id,device,10**9)
        hidden=teacher.model(ids,attention_mask=attention,use_cache=False).last_hidden_state
        for index,row in enumerate(part):
            count=min(32,len(row['response_ids']))
            positions=torch.linspace(len(row['prompt_ids'])-1,len(row['prompt_ids'])+len(row['response_ids'])-2,count,device=device).long()
            logits=teacher.lm_head(hidden[index,positions]).float()
            all_entropy=[]
            for scoring_temperature in [.6,1.,1.4]:
                logp=(logits/scoring_temperature).log_softmax(-1)
                entropy=-(logp.exp()*logp).sum(-1)
                all_entropy.append(entropy)
                for position,value in zip(positions.cpu().tolist(),entropy.cpu().tolist(),strict=True):
                    records.append({'trajectory_temperature':args.temperature,'scoring_temperature':scoring_temperature,
                                    'prompt_id':row['prompt_id'],'position':position,'entropy_nats':value})
            assert bool((all_entropy[0]<=all_entropy[1]+1e-5).all())
            assert bool((all_entropy[1]<=all_entropy[2]+1e-5).all())
        if start%256==0:
            print('entropy prompts',start,flush=True)
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    frame=pl.DataFrame(records)
    frame.write_parquet(output)
    summary=frame.group_by(['trajectory_temperature','scoring_temperature']).agg(
        pl.len().alias('positions'),pl.col('entropy_nats').mean().alias('mean_nats'),
        pl.col('entropy_nats').quantile(.1).alias('p10_nats'),
        pl.col('entropy_nats').median().alias('median_nats'),
        pl.col('entropy_nats').quantile(.9).alias('p90_nats'),
        (pl.col('entropy_nats')<.1).mean().alias('fraction_below_0_1_nats')).sort(['trajectory_temperature','scoring_temperature'])
    summary.write_json(output.with_suffix('.json'))
    print(summary,flush=True)


if __name__=='__main__':
    main(Args().parse_args())
