"""Check plateau decisions, stored-token alignment and real AMP loss gradients."""

import json

import torch

from examples.experimental.spec_draft_transition.adaptive import PlateauTransition
from examples.experimental.spec_draft_transition.loss import probability_loss
from examples.experimental.spec_draft_transition.train import batch, teacher_labels


def main() -> None:
    flat = PlateauTransition()
    for step in range(0, 600, 100):
        decision = flat.observe(step, 0.8)
        assert decision['triggered'] == (step == 500)
    assert flat.weights() == (0.0, 0.1, 0.9)
    assert not flat.observe(600, 0.7)['triggered']
    improving = PlateauTransition()
    for step in range(0, 2100, 100):
        improving.observe(step, 1.0 * 0.98 ** (step // 100))
    assert improving.switch_step is None
    plateau = PlateauTransition()
    for step, value in zip(range(0, 900, 100), [1, .9, .8, .7, .6, .5, .501, .499, .498]):
        plateau.observe(step, value)
    assert plateau.switch_step == 800
    device = torch.device('cuda')
    rows = [{'prompt_ids':[3,4], 'response_ids':[5,6,7]},
            {'prompt_ids':[8], 'response_ids':[9,10]}]
    ids, attention, mask = batch(rows, 0, device, 4)
    assert teacher_labels(rows, mask).tolist() == [5,6,7,9]
    torch.manual_seed(812)
    hidden = torch.randn(5, 7, device=device, dtype=torch.bfloat16)
    th = torch.randn(5, 9, device=device, dtype=torch.bfloat16)
    tw = torch.randn(13, 9, device=device, dtype=torch.bfloat16)
    labels = torch.tensor([1,8,3,11,5], device=device)
    errors = []
    for hard, soft, l1 in [(0.1,0.9,0.0),(0.0,0.1,0.9)]:
        weights = torch.randn(13,7,device=device,dtype=torch.float32,requires_grad=True)
        h = hidden.detach().clone().requires_grad_(True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            actual = probability_loss(h,weights,th,tw,1.4,soft,5,99,labels,hard,l1)
            logq = (h @ weights.T).float().log_softmax(-1)
            p = ((th @ tw.T).float()/1.4).softmax(-1)
            expected = (soft * -(p*logq).sum() + l1*(p-logq.exp()).abs().sum()
                        + hard * -logq.gather(-1,labels[:,None]).sum()) / 5
        ga = torch.autograd.grad(actual,(h,weights),retain_graph=True)
        ge = torch.autograd.grad(expected,(h,weights))
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)
        for a,e in zip(ga,ge):
            torch.testing.assert_close(a,e,atol=0.005,rtol=0.03)
        errors.append({'loss_abs_error':float((actual-expected).detach().abs()),
                       'gradient_max_abs_error':max(float((a-e).abs().max()) for a,e in zip(ga,ge))})
    print(json.dumps({'passed':True,'plateau_switch':800,'amp_checks':errors}))


if __name__ == '__main__':
    main()
