"""Reference Hugging Face throughput with equal output caps and EOS handling.

This measures this implementation, not an optimized production inference engine.
Compare teacher-only and speculative decoding on the same GPU and prompt set.
"""

import json
import time
from pathlib import Path

import torch
from tap import Tap
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.experimental.spec_draft_transition.evaluate import probabilities, sample_sequence
from examples.experimental.spec_draft_transition.train import load_rows


class Args(Tap):
    teacher: str
    student: str
    data: str
    output: str
    temperature: float
    prompts: int = 128
    repetitions: int = 3
    repetition_start: int = 0
    max_tokens: int = 256
    seed: int = 22931


@torch.no_grad()
def teacher_sequence(teacher, prompt: list[int], temperature: float, eos: set[int], maximum: int) -> int:
    device=next(teacher.parameters()).device
    cache=teacher.model(torch.tensor([prompt[:-1]],device=device),use_cache=True).past_key_values if len(prompt)>1 else None
    ids=torch.tensor([[prompt[-1]]],device=device)
    for count in range(1,maximum+1):
        result=teacher(ids,past_key_values=cache,use_cache=True)
        cache=result.past_key_values
        token=torch.multinomial(probabilities(result.logits[0,-1],temperature),1)
        ids=token[None,:]
        if int(token) in eos:
            return count
    return maximum


def main(args: Args) -> None:
    torch.set_num_threads(4)
    device=torch.accelerator.current_accelerator()
    teacher=AutoModelForCausalLM.from_pretrained(args.teacher,dtype=torch.bfloat16,attn_implementation='sdpa').to(device).eval()
    student=AutoModelForCausalLM.from_pretrained(args.student,dtype=torch.bfloat16,attn_implementation='sdpa').to(device).eval()
    eos=teacher.generation_config.eos_token_id
    eos=set(eos if isinstance(eos,list) else [eos])
    rows=load_rows(args.data,args.temperature,'test')[:args.prompts]
    for row in rows[:4]:
        teacher_sequence(teacher,row['prompt_ids'],args.temperature,eos,32)
        sample_sequence(teacher,student,row['prompt_ids'],args.temperature,eos,8,32,32)
    results=[]
    for repetition in range(args.repetition_start,args.repetition_start+args.repetitions):
        order=['teacher','speculative'] if repetition%2==0 else ['speculative','teacher']
        for mode in order:
            torch.manual_seed(args.seed+repetition)
            torch.accelerator.synchronize()
            started=time.perf_counter()
            tokens=0
            for row in rows:
                if mode=='teacher':
                    tokens+=teacher_sequence(teacher,row['prompt_ids'],args.temperature,eos,args.max_tokens)
                else:
                    result=sample_sequence(teacher,student,row['prompt_ids'],args.temperature,eos,8,args.max_tokens,args.max_tokens)
                    tokens+=len(result['generated_ids'])
                    assert len(result['generated_ids'])<=args.max_tokens
            torch.accelerator.synchronize()
            seconds=time.perf_counter()-started
            record={'mode':mode,'repetition':repetition,'tokens':tokens,'seconds':seconds,'tokens_per_second':tokens/seconds}
            results.append(record)
            print(json.dumps(record),flush=True)
    Path(args.output).write_text(json.dumps({'config':args.as_dict(),'results':results,
        'scope':'Single-request Hugging Face reference implementation; same GPU, EOS and output cap; model loading excluded.'},indent=2))


if __name__=='__main__':
    main(Args().parse_args())
