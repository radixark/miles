"""Executable numerical and real-model preflight; no training is launched."""

import json
from pathlib import Path

import torch
from tap import Tap
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.experimental.spec_draft_transition.loss import ProbabilityLoss, ce_weight, probability_metrics


class Args(Tap):
    teacher: str
    student: str
    output: str


def verify_gradients() -> dict[str, float]:
    torch.manual_seed(736)
    maximum = 0.0
    for temperature in [0.6, 1.0, 1.4]:
        for mix in [0.0, 0.37, 1.0]:
            hidden = torch.randn(11, 7, requires_grad=True)
            weight = torch.randn(19, 7, requires_grad=True)
            teacher_hidden, teacher_weight = torch.randn(11, 9), torch.randn(19, 9)
            actual = ProbabilityLoss.apply(hidden, weight, teacher_hidden, teacher_weight, temperature, mix, 17, 3)
            actual_grads = torch.autograd.grad(actual, (hidden, weight))
            logq = (hidden @ weight.T).log_softmax(-1)
            p = ((teacher_hidden @ teacher_weight.T) / temperature).softmax(-1)
            expected = (mix * -(p * logq).sum() + (1 - mix) * (p - logq.exp()).abs().sum()) / 17
            expected_grads = torch.autograd.grad(expected, (hidden, weight))
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
            for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
                torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=2e-5)
                maximum = max(maximum, float((actual_grad - expected_grad).abs().max()))
    assert ce_weight('smooth', 0, 2000) == 1.0
    assert ce_weight('smooth', 1999, 2000) == 0.0
    return {'maximum_gradient_error': maximum}


def main(args: Args) -> None:
    report = verify_gradients()
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    student_tokenizer = AutoTokenizer.from_pretrained(args.student)
    assert teacher_tokenizer.get_vocab() == student_tokenizer.get_vocab(), 'token IDs differ'
    assert teacher_tokenizer.eos_token_id == student_tokenizer.eos_token_id
    device = torch.accelerator.current_accelerator()
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=torch.bfloat16, attn_implementation='sdpa').to(device).eval()
    student = AutoModelForCausalLM.from_pretrained(args.student, dtype=torch.bfloat16, attn_implementation='sdpa').to(device).eval()
    tokens = teacher_tokenizer.apply_chat_template([{'role':'user','content':'Explain why the sky appears blue in two sentences.'}], enable_thinking=False, add_generation_prompt=True, return_tensors='pt').to(device)
    with torch.no_grad():
        generated = teacher.generate(tokens, max_new_tokens=64, do_sample=True, temperature=1.0, top_p=1.0, top_k=0)
        th = teacher.model(generated[:, :-1], use_cache=False).last_hidden_state.flatten(0,1)
        sh = student.model(generated[:, :-1], use_cache=False).last_hidden_state.flatten(0,1)
        report['metrics'] = {str(t): probability_metrics(sh, student.lm_head.weight, th, teacher.lm_head.weight, t) for t in [0.6,1.0,1.4]}
        entropies = [report['metrics'][str(t)]['entropy'] for t in [0.6,1.0,1.4]]
        assert entropies[0] < entropies[1] < entropies[2]
    report['vocabulary_size'] = len(teacher_tokenizer)
    report['status'] = 'passed'
    Path(args.output).write_text(json.dumps(report,indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main(Args().parse_args())
