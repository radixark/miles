"""Single-GPU, full-parameter drafter distillation on a frozen teacher corpus.

This independent worker avoids distributed collectives so independent loss,
temperature and seed combinations can occupy separate GPUs concurrently.
"""

import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
import wandb
from tap import Tap
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.experimental.spec_draft_transition.loss import probability_loss, ce_weight, probability_metrics
from examples.experimental.spec_draft_transition.adaptive import PlateauTransition


class Args(Tap):
    teacher: str
    student: str
    data: str
    output: str
    run_id: str
    temperature: float
    method: str
    learning_rate: float
    seed: int
    updates: int = 2000
    tokens_per_update: int = 16384
    micro_batch: int = 2
    weight_decay: float = 0.01
    warmup: int = 100
    validation_interval: int = 200
    validation_tokens: int = 4096
    chunk: int = 1024
    wandb_entity: str = 'radixarkai'
    wandb_project: str = 'spec-draft-transition'
    wandb_mode: str = 'online'


def load_rows(data: str, temperature: float, split: str) -> list[dict]:
    rows = []
    for path in sorted(Path(data).glob(f't{temperature}-shard*.jsonl')):
        assert path.with_suffix('.done').exists(), path
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                if row['split'] == split:
                    rows.append(row)
    if not rows:
        raise ValueError(f'No {split} rows for temperature {temperature}')
    rows.sort(key=lambda row: row['prompt_id'])
    return rows


def batch(rows: list[dict], pad: int, device: torch.device, remaining: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    length = max(len(r['prompt_ids']) + len(r['response_ids']) - 1 for r in rows)
    ids = torch.full((len(rows), length), pad, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for i, row in enumerate(rows):
        tokens = row['prompt_ids'] + row['response_ids']
        end = len(tokens) - 1
        ids[i, :end] = torch.tensor(tokens[:-1], device=device)
        attention[i, :end] = 1
        start = len(row['prompt_ids']) - 1
        count = min(end - start, remaining)
        mask[i, start:start + count] = True
        remaining -= count
    return ids, attention, mask


def teacher_labels(rows: list[dict], mask: torch.Tensor) -> torch.Tensor:
    """At input position j, the stored teacher draw is concatenated_tokens[j+1]."""
    targets = torch.zeros_like(mask, dtype=torch.long)
    for index, row in enumerate(rows):
        tokens = row['prompt_ids'] + row['response_ids']
        targets[index, :len(tokens) - 1] = torch.tensor(tokens[1:], device=mask.device)
    return targets[mask]


def hidden_states(model, ids: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
    return model.model(input_ids=ids, attention_mask=attention, use_cache=False).last_hidden_state


@torch.no_grad()
def validate(student, teacher, rows: list[dict], tokenizer, args: Args) -> dict[str, float]:
    student.eval()
    totals = dict(ce=0.0, kl=0.0, l1=0.0, entropy=0.0, tokens=0.0)
    for start in range(0, len(rows), args.micro_batch):
        remaining = args.validation_tokens - int(totals['tokens'])
        if remaining <= 0:
            break
        ids, attention, mask = batch(rows[start:start + args.micro_batch], tokenizer.pad_token_id,
                                     next(student.parameters()).device, remaining)
        # Spread the fitting curves over many conversations instead of letting
        # a handful of long responses exhaust the validation token budget.
        for index in range(len(mask)):
            positions = mask[index].nonzero().flatten()
            keep = positions[torch.linspace(0, len(positions)-1, min(32,len(positions)),device=mask.device).long()]
            mask[index] = False
            mask[index,keep] = True
        th = hidden_states(teacher, ids, attention)[mask].to(torch.bfloat16)
        sh = hidden_states(student, ids, attention)[mask].to(torch.bfloat16)
        metrics = probability_metrics(sh, student.lm_head.weight, th, teacher.lm_head.weight, args.temperature, args.chunk)
        for key, value in metrics.items():
            totals[key] += value
    count = totals.pop('tokens')
    result = {key: value / count for key, value in totals.items()}
    result['overlap'] = 1 - result['l1'] / 2
    result['tokens'] = count
    student.train()
    return result


def save_checkpoint(student, tokenizer, output: Path, update: int) -> None:
    destination = output / f'checkpoint-{update:04d}'
    student.save_pretrained(destination)
    tokenizer.save_pretrained(destination)


def train_update(student, teacher, optimizer, rows: list[dict], tokenizer, args: Args,
                 rng: random.Random, update: int, transition: PlateauTransition | None = None) -> dict[str, float]:
    warmup = min(1.0, (update + 1) / args.warmup)
    progress = max(0.0, (update - args.warmup) / max(1, args.updates - args.warmup - 1))
    lr = args.learning_rate * warmup * (0.01 + 0.99 * (1 + math.cos(math.pi * progress)) / 2)
    for group in optimizer.param_groups:
        group['lr'] = lr
    hard, mix, l1_coefficient = transition.weights() if transition else (0.0, ce_weight(args.method, update, args.updates), None)
    optimizer.zero_grad(set_to_none=True)
    remaining, total = args.tokens_per_update, 0.0
    while remaining:
        selected = [rows[rng.randrange(len(rows))] for _ in range(args.micro_batch)]
        ids, attention, mask = batch(selected, tokenizer.pad_token_id, next(student.parameters()).device, remaining)
        count = int(mask.sum())
        with torch.no_grad():
            th = hidden_states(teacher, ids, attention)[mask].to(torch.bfloat16)
        sh = hidden_states(student, ids, attention)[mask].to(torch.bfloat16)
        loss = probability_loss(sh, student.lm_head.weight, th, teacher.lm_head.weight,
                                     args.temperature, mix, args.tokens_per_update, args.chunk,
                                     labels=teacher_labels(selected, mask) if hard else None,
                                     hard_weight=hard, l1_weight=l1_coefficient)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'nonfinite loss update={update}')
        loss.backward()
        total += float(loss.detach())
        remaining -= count
    gradient = torch.nn.utils.clip_grad_norm_(student.parameters(), float('inf'))
    if not torch.isfinite(gradient):
        raise FloatingPointError(f'nonfinite gradient update={update}')
    optimizer.step()
    return dict(loss=total, ce_weight=mix, hard_ce_weight=hard,
                l1_weight=1-mix if l1_coefficient is None else l1_coefficient,
                learning_rate=lr, gradient_norm=float(gradient))


def main(args: Args) -> None:
    if args.method == 'adaptive':
        args.validation_interval = 100
    if os.environ.get('WANDB_API_KEY_FILE'):
        os.environ['WANDB_API_KEY'] = Path(os.environ['WANDB_API_KEY_FILE']).read_text().strip()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = Path(__file__).with_name('manifest.json')
    manifest = json.loads(manifest_path.read_text())
    entry = next((job for job in manifest['jobs'] if job['run_id']==args.run_id),None)
    tuning_seeds = {job['seed'] for job in manifest['jobs'] if job['stage']=='tuning'}
    stage = entry['stage'] if entry else 'tuning' if args.seed in tuning_seeds else 'preflight'
    revision_file = Path(__file__).resolve().parents[3]/'SOURCE_REVISION'
    configuration = {**args.as_dict(), 'stage':stage, 'argv':sys.argv,
                     'source_revision':revision_file.read_text().strip() if revision_file.exists() else 'unrecorded'}
    (output / 'config.json').write_text(json.dumps(configuration, indent=2, default=str))
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    other = AutoTokenizer.from_pretrained(args.student)
    assert tokenizer.get_vocab() == other.get_vocab()
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=torch.bfloat16, attn_implementation='sdpa').to(torch.accelerator.current_accelerator()).eval()
    teacher.requires_grad_(False)
    student = AutoModelForCausalLM.from_pretrained(args.student, dtype=torch.bfloat16, attn_implementation='sdpa').to(torch.accelerator.current_accelerator()).train()
    # Keep trainable parameters and optimizer moments in float32; autocast is
    # applied only to model operations, avoiding bf16 Adam moment accumulation.
    student.float()
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate, betas=(0.9,0.999), eps=1e-8,
                                 weight_decay=args.weight_decay, fused=True)
    train_rows = load_rows(args.data, args.temperature, 'train')
    validation_rows = load_rows(args.data, args.temperature, 'validation')
    run = wandb.init(entity=args.wandb_entity, project=args.wandb_project, id=args.run_id,
                     name=f"{datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%y%m%d')}-{args.method}-t{args.temperature}-s{args.seed}-{args.run_id.rsplit('-',1)[-1]}",
                     config=configuration, group=manifest['experiment_id'], job_type=stage,
                     mode=args.wandb_mode, dir=str(output))
    (output / 'wandb-url.txt').write_text(str(run.url))
    rng = random.Random(args.seed)
    transition = PlateauTransition() if args.method == 'adaptive' else None
    started = time.monotonic()
    with (output / 'metrics.jsonl').open('w') as handle:
        for update in range(args.updates + 1):
            if update % args.validation_interval == 0 or update == args.updates:
                with torch.autocast(device_type=torch.accelerator.current_accelerator().type, dtype=torch.bfloat16):
                    metrics = validate(student, teacher, validation_rows, tokenizer, args)
                record = {'update':update, 'kind':'validation', **metrics}
                handle.write(json.dumps(record) + '\n'); handle.flush()
                run.log({f'validation/{key}':value for key,value in metrics.items()}, step=update, commit=False)
                print(json.dumps(record), flush=True)
                if transition is not None and update % transition.interval == 0:
                    decision = transition.observe(update, metrics['l1'], allow_switch=update < args.updates)
                    handle.write(json.dumps({'kind':'transition_check', **decision}) + '\n')
                    handle.flush()
                    (output / 'transition.json').write_text(json.dumps({
                        **decision, 'history':transition.history,
                        'weights':transition.weights(),
                        'interval':transition.interval, 'window':transition.window,
                        'minimum_updates':transition.minimum_updates,
                        'relative_threshold':transition.relative_threshold}, indent=2))
                    run.log({'transition/switched':int(transition.switch_step is not None),
                             'transition/relative_improvement':decision['window_relative_best_improvement'],
                             'transition/switch_step':transition.switch_step}, step=update, commit=False)
                    print(json.dumps({'kind':'transition_check', **decision}), flush=True)
                if update and (update == args.updates or update % 500 == 0):
                    save_checkpoint(student, tokenizer, output, update)
            if update == args.updates:
                break
            with torch.autocast(device_type=torch.accelerator.current_accelerator().type, dtype=torch.bfloat16):
                metrics = train_update(student, teacher, optimizer, train_rows, tokenizer, args, rng, update, transition)
            record = {'update':update + 1, 'kind':'training', 'seconds':time.monotonic() - started, **metrics}
            handle.write(json.dumps(record) + '\n'); handle.flush()
            run.log({f'train/{key}':value for key,value in metrics.items()}, step=update + 1, commit=False)
            print(json.dumps(record), flush=True)
    (output / 'complete.json').write_text(json.dumps({'seconds':time.monotonic()-started,'updates':args.updates,
        'peak_cuda_memory_bytes':torch.cuda.max_memory_allocated()}))
    run.finish()


if __name__ == '__main__':
    main(Args().parse_args())
