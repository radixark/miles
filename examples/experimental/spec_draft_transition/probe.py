"""Executable numerical and real-model preflight; no training is launched."""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from tap import Tap
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.experimental.spec_draft_transition.loss import probability_loss, ce_weight, probability_metrics
from examples.experimental.spec_draft_transition.evaluate import expected_prefix, sample_sequence
from examples.experimental.spec_draft_transition.pipeline import State


class Args(Tap):
    teacher: str
    student: str
    output: str


class ConstantCache:
    def __init__(self, length: int = 0) -> None:
        self.length = length

    def crop(self, length: int) -> None:
        assert length <= self.length
        self.length = length


class ConstantModel:
    def __init__(self, probabilities: list[float]) -> None:
        self.logits = torch.tensor(probabilities).log()
        self.parameter = torch.nn.Parameter(torch.zeros(()))
        self.model = self
        self.lm_head = torch.nn.Identity()
        self.generation_config = SimpleNamespace(eos_token_id=1)

    def parameters(self):
        yield self.parameter

    def __call__(self, input_ids, past_key_values=None, use_cache=True, **kwargs):
        ids = input_ids
        cache = past_key_values or ConstantCache()
        cache.length += ids.shape[1]
        logits=self.logits[None,None,:].expand(ids.shape[0],ids.shape[1],-1)
        return SimpleNamespace(past_key_values=cache,logits=logits,last_hidden_state=logits)


def verify_sampler() -> dict[str, float]:
    row={'prompt_ids':[0,0],'response_ids':[1],'prompt_id':'0'*64}
    eos_only=expected_prefix(ConstantModel([0.,1.]),ConstantModel([0.,1.]),[row],SimpleNamespace(pad_token_id=0),1.,1,1,8)
    assert eos_only['expected_prefix']==1.,eos_only
    identical = sample_sequence(ConstantModel([.2,.8]),ConstantModel([.2,.8]),[0,1],1.,set(),8,3)
    assert identical['accepted']==[8,8,8] and identical['emitted']==[9,9,9]
    disjoint = sample_sequence(ConstantModel([1.,0.]),ConstantModel([0.,1.]),[0,1],1.,set(),8,3)
    assert disjoint['accepted']==[0,0,0] and disjoint['generated_ids']==[0,0,0]
    torch.manual_seed(173)
    teacher, student = ConstantModel([.8,.2]),ConstantModel([.2,.8])
    values = [sample_sequence(teacher,student,[0,1],1.,set(),8,1) for _ in range(5000)]
    token_zero = sum(row['generated_ids'][0]==0 for row in values)/len(values)
    accepted = sum(row['accepted'][0] for row in values)/len(values)
    expected = sum(.4**i for i in range(1,9))
    assert abs(token_zero-.8)<.025 and abs(accepted-expected)<.04
    return {'first_token_probability':token_zero,'expected_probability':.8,
            'accepted_prefix':accepted,'expected_prefix':expected}


def verify_scheduler() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        jobs=[{'run_id':str(i),'stage':'tuning','status':'pending','temperature':1.,'method':'ce','learning_rate':rate}
              for i,rate in enumerate([1e-5,3e-5,1e-4])]
        jobs.append({'run_id':'confirm','stage':'confirmation','status':'pending','temperature':1.,'method':'ce','learning_rate':None})
        manifest=root/'manifest.json'
        manifest.write_text(json.dumps({'jobs':jobs,'temperatures':[1.],'methods':['ce'],'shards':1}))
        state=State(SimpleNamespace(root=str(root),manifest=str(manifest)))
        state.action('/generation',{'slot':0})
        for score in [1.,2.,1.]:
            job=state.action('/claim',{'slot':0})['job']
            state.action('/complete',{'run_id':job['run_id'],'success':True,'validation':{'expected_prefix':score}})
        assert state.state['phase']=='confirmation'
        assert state.action('/claim',{'slot':0})['job']['learning_rate']==3e-5


def verify_gradients() -> dict[str, float]:
    torch.manual_seed(736)
    maximum = 0.0
    for temperature in [0.6, 1.0, 1.4]:
        for mix in [0.0, 0.37, 1.0]:
            hidden = torch.randn(11, 7, requires_grad=True)
            weight = torch.randn(19, 7, requires_grad=True)
            teacher_hidden, teacher_weight = torch.randn(11, 9), torch.randn(19, 9)
            actual = probability_loss(hidden, weight, teacher_hidden, teacher_weight, temperature, mix, 17, 3)
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
    report['sampler'] = verify_sampler()
    verify_scheduler()
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    student_tokenizer = AutoTokenizer.from_pretrained(args.student)
    assert teacher_tokenizer.get_vocab() == student_tokenizer.get_vocab(), 'token IDs differ'
    assert teacher_tokenizer.eos_token_id == student_tokenizer.eos_token_id
    device = torch.accelerator.current_accelerator()
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=torch.bfloat16, attn_implementation='sdpa').to(device).eval()
    student = AutoModelForCausalLM.from_pretrained(args.student, dtype=torch.bfloat16, attn_implementation='sdpa').to(device).eval()
    tokens = teacher_tokenizer.apply_chat_template([{'role':'user','content':'Explain why the sky appears blue in two sentences.'}], enable_thinking=False, add_generation_prompt=True, return_dict=False, return_tensors='pt').to(device)
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
