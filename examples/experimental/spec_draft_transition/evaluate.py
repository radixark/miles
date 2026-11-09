"""Correct sequential speculative sampling and validation path integration."""

import json
import time
from pathlib import Path

import torch
from tap import Tap
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.experimental.spec_draft_transition.train import load_rows


class Args(Tap):
    teacher: str
    student: str
    data: str
    output: str
    temperature: float
    split: str = 'test'
    prompts: int = 2000
    repetitions: int = 1
    rounds: int = 32
    draft: int = 8
    seed: int = 71932
    expected: bool = False


def probabilities(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    return (logits.float() / temperature).softmax(-1)


@torch.no_grad()
def expected_prefix(teacher, student, rows: list[dict], tokenizer, temperature: float,
                    repetitions: int = 4, limit: int = 128, draft: int = 8) -> dict[str, float]:
    device = next(student.parameters()).device
    values = []
    prefixes = []
    for row in rows[:limit]:
        cut = int(row['prompt_id'][:8], 16) % len(row['response_ids'])
        cut = min(cut, 2048 - draft - len(row['prompt_ids']))
        prefixes.append({**row, 'prompt_ids':row['prompt_ids'] + row['response_ids'][:max(0,cut)]})
    records = prefixes * repetitions
    for start in range(0, len(records), 8):
        part = records[start:start + 8]
        width = max(len(row['prompt_ids']) for row in part)
        ids = torch.full((len(part),width),tokenizer.pad_token_id,device=device,dtype=torch.long)
        attention = torch.zeros_like(ids)
        for i,row in enumerate(part):
            n = len(row['prompt_ids'])
            ids[i,-n:] = torch.tensor(row['prompt_ids'],device=device)
            attention[i,-n:] = 1
        original = ids
        cache, sampled, q_values = None, [], []
        for step in range(draft):
            positions = attention.cumsum(-1) - 1
            positions.masked_fill_(attention == 0, 0)
            result = student(input_ids=ids, attention_mask=attention, position_ids=positions[:,-ids.shape[1]:],
                             past_key_values=cache, use_cache=True)
            cache = result.past_key_values
            q = probabilities(result.logits[:,-1])
            token = torch.multinomial(q,1)
            sampled.append(token)
            q_values.append(q.gather(-1,token))
            ids = token
            attention = torch.cat([attention,torch.ones_like(token)],-1)
        proposal = torch.cat(sampled,-1)
        verify_ids = torch.cat([original,proposal[:,:-1]],-1)
        verify_attention = attention[:,:-1]
        positions = verify_attention.cumsum(-1)-1
        positions.masked_fill_(verify_attention==0,0)
        hidden = teacher.model(verify_ids,attention_mask=verify_attention,position_ids=positions,use_cache=False).last_hidden_state[:,width-1:]
        p = probabilities(teacher.lm_head(hidden),temperature)
        ratios = p.gather(-1,proposal.unsqueeze(-1)).squeeze(-1) / torch.cat(q_values,-1)
        expected = ratios.clamp(max=1).cumprod(-1).sum(-1)
        values.extend(expected.cpu().tolist())
    return {'expected_prefix':sum(values)/len(values),'paths':len(values)}


@torch.no_grad()
def sample_sequence(teacher, student, prompt: list[int], temperature: float, eos: set[int],
                    draft: int, rounds: int) -> dict:
    device = next(student.parameters()).device
    history = list(prompt)
    teacher_cache, student_cache = None, None
    # Caches represent the history excluding its last token at each block start.
    if len(prompt) > 1:
        prefix = torch.tensor([prompt[:-1]],device=device)
        teacher_cache = teacher.model(prefix,use_cache=True).past_key_values
        student_cache = student.model(prefix,use_cache=True).past_key_values
    accepted_lengths, emitted_lengths, corrections, bonuses = [], [], 0, 0
    for block in range(rounds):
        old_length = len(history)
        proposals, qs = [], []
        token = history[-1]
        for _ in range(draft):
            output = student(torch.tensor([[token]],device=device),past_key_values=student_cache,use_cache=True)
            student_cache = output.past_key_values
            q = probabilities(output.logits[0,-1])
            token = int(torch.multinomial(q,1))
            proposals.append(token); qs.append(q)
            if token in eos:
                break
        inputs = torch.tensor([[history[-1]] + proposals],device=device)
        output = teacher(inputs,past_key_values=teacher_cache,use_cache=True)
        teacher_cache = output.past_key_values
        ps = probabilities(output.logits[0],temperature)
        accepted, extra, finished = 0, None, False
        for j, token in enumerate(proposals):
            alpha = min(1.0,float(ps[j,token]/qs[j][token]))
            if float(torch.rand((),device=device)) < alpha:
                accepted += 1
                if token in eos:
                    finished = True
                    break
            else:
                residual = (ps[j]-qs[j]).clamp_min(0)
                if not float(residual.sum()) > 0:
                    raise FloatingPointError('empty rejection residual')
                extra = int(torch.multinomial(residual,1))
                corrections += 1
                break
        if not finished and extra is None:
            extra = int(torch.multinomial(ps[len(proposals)],1))
            bonuses += 1
            # The drafting cache has not consumed the final accepted proposal.
            student_cache = student.model(torch.tensor([[proposals[-1]]],device=device),past_key_values=student_cache,use_cache=True).past_key_values
        history.extend(proposals[:accepted])
        if extra is not None:
            history.append(extra)
        accepted_lengths.append(accepted)
        emitted_lengths.append(accepted + int(extra is not None))
        if finished or extra in eos:
            break
        teacher_cache.crop(old_length+accepted)
        student_cache.crop(old_length+accepted)
    return {'accepted':accepted_lengths,'emitted':emitted_lengths,'corrections':corrections,'bonuses':bonuses,
            'generated_ids':history[len(prompt):]}


def main(args: Args) -> None:
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.accelerator.current_accelerator()
    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher,dtype=torch.bfloat16,attn_implementation='sdpa').to(device).eval()
    student = AutoModelForCausalLM.from_pretrained(args.student,dtype=torch.bfloat16,attn_implementation='sdpa').to(device).eval()
    rows = load_rows(args.data,args.temperature,args.split)
    output = Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    if args.expected:
        report = expected_prefix(teacher,student,rows,tokenizer,args.temperature,args.repetitions,args.prompts,args.draft)
        output.write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
        return
    eos = set(teacher.generation_config.eos_token_id if isinstance(teacher.generation_config.eos_token_id,list) else [teacher.generation_config.eos_token_id])
    started = time.monotonic()
    with output.open('w') as handle:
        for row in rows[:args.prompts]:
            for repetition in range(args.repetitions):
                result = sample_sequence(teacher,student,row['prompt_ids'],args.temperature,eos,args.draft,args.rounds)
                result.update(prompt_id=row['prompt_id'],repetition=repetition)
                handle.write(json.dumps(result)+'\n');handle.flush()
            print(json.dumps({'prompt':row['prompt_id'],'seconds':time.monotonic()-started}),flush=True)


if __name__ == '__main__':
    main(Args().parse_args())
