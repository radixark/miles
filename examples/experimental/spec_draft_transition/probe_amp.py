"""Compare custom gradients with eager autograd in the actual BF16 training path."""

import json
from pathlib import Path

import torch
from tap import Tap
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.experimental.spec_draft_transition.loss import ProbabilityLoss


class Args(Tap):
    teacher: str
    student: str
    output: str


def reference_loss(hidden: torch.Tensor, weight: torch.Tensor, teacher_hidden: torch.Tensor,
                   teacher_weight: torch.Tensor, temperature: float, mix: float, chunk: int) -> torch.Tensor:
    total = hidden.new_zeros((), dtype=torch.float32)
    for h, th in zip(hidden.split(chunk), teacher_hidden.split(chunk), strict=True):
        logq = (h @ weight.T).float().log_softmax(-1)
        p = ((th @ teacher_weight.T).float() / temperature).softmax(-1)
        total = total + mix * -(p * logq).sum() + (1 - mix) * (p - logq.exp()).abs().sum()
    return total / len(hidden)


def compare(hidden: torch.Tensor, weight: torch.Tensor, teacher_hidden: torch.Tensor,
            teacher_weight: torch.Tensor, temperature: float, mix: float, chunk: int) -> dict:
    with torch.autocast('cuda', dtype=torch.bfloat16):
        custom = ProbabilityLoss.apply(hidden, weight, teacher_hidden, teacher_weight, temperature, mix, len(hidden), chunk)
        reference = reference_loss(hidden, weight, teacher_hidden, teacher_weight, temperature, mix, chunk)
    actual = torch.autograd.grad(custom, (hidden, weight))
    expected = torch.autograd.grad(reference, (hidden, weight))
    torch.testing.assert_close(custom, reference, atol=2e-6, rtol=2e-5)
    errors = []
    for original, a, b in zip([hidden, weight], actual, expected, strict=True):
        assert a.dtype == original.dtype
        relative = float((a - b).norm() / b.norm().clamp_min(1e-5))
        absolute = float((a - b).abs().max())
        assert relative < 0.02, (temperature, mix, relative, absolute)
        errors.append({'relative_l2': relative, 'maximum_absolute': absolute})
    return {'temperature': temperature, 'ce_weight': mix, 'loss': float(custom.detach()), 'gradient_errors': errors}


def main(args: Args) -> None:
    torch.manual_seed(824)
    torch.set_num_threads(2)
    results = []
    for matching in [False, True]:
        for temperature in [0.6, 1.0, 1.4]:
            for mix in [0.0, 0.37, 1.0]:
                h = torch.randn(11, 7, device='cuda', requires_grad=True)
                w = torch.randn(19, 7, device='cuda', requires_grad=True)
                th = h.detach().to(torch.bfloat16) if matching else torch.randn(11, 7, device='cuda', dtype=torch.bfloat16)
                tw = w.detach().to(torch.bfloat16) if matching else torch.randn(19, 7, device='cuda', dtype=torch.bfloat16)
                results.append({'matching': matching, **compare(h, w, th, tw, temperature, mix, 3)})
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
    student = AutoModelForCausalLM.from_pretrained(args.student, dtype=torch.float32, attn_implementation='sdpa').cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    tokens = tokenizer.apply_chat_template([{'role': 'user', 'content': 'Explain why the sky is blue.'}],
        enable_thinking=False, add_generation_prompt=True, return_dict=False, return_tensors='pt').cuda()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        sh = student.model(tokens, use_cache=False).last_hidden_state[0, -11:]
        th = teacher.model(tokens, use_cache=False).last_hidden_state[0, -11:]
    assert sh.dtype == torch.float32 and th.dtype == torch.bfloat16
    for mix in [0.0, 0.37, 1.0]:
        results.append({'real_model': True, **compare(sh.detach().requires_grad_(), student.lm_head.weight,
            th, teacher.lm_head.weight, 1.0, mix, 3)})
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(json.dumps({'amp_cases': len(results), 'maximum_relative_l2': max(e['relative_l2'] for r in results for e in r['gradient_errors']),
                      'actual_student_hidden_dtype': str(sh.dtype), 'status': 'passed'}), flush=True)


if __name__ == '__main__':
    main(Args().parse_args())
