"""A trie-packed micro-batch must score every sample as if it ran alone, on the real flash-attn kernels, and train
on gradients as accurate as standard packing's.

The CPU tests check the layout against a plain-PyTorch reference; this runs the FSDP actor's own input
builder and logits path on tiny models with the real kernels:

- Qwen3 on flash-attn, with and without a sliding-window layer
"""

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=300, suite="stage-b-2-gpu-h200", labels=["fsdp"], hardware=["hopper"], num_gpus=1)

import contextlib
import inspect
import random
import sys
from statistics import geometric_mean
from types import SimpleNamespace

import pytest
import torch
from tests.fast.backends.training_utils.reference_trie_attention import reference_trie_attention
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from miles.backends.fsdp_utils import actor as fsdp_actor
from miles.backends.fsdp_utils.actor import FSDPTrainRayActor
from miles.backends.fsdp_utils.adaptations.specs import resolve_arch_adapter
from miles.backends.fsdp_utils.adaptations.trie_attention import (
    FLASH_ATTN_MODULES,
    TrieAttentionArgs,
    register_trie_attention,
    trie_attention_forward,
    trie_attn_implementation,
    trie_model_spec,
)
from miles.backends.fsdp_utils.adaptations.trie_gather import gather_rows
from miles.backends.training_utils.data import context_parallel
from miles.backends.training_utils.data import rollout as data_utils
from miles.backends.training_utils.data.trie_batch import TrieMicroBatches, trie_packing_keys
from miles.backends.training_utils.data.trie_packing import WINDOW_START_MULTIPLE, build_trie_layout, pack_trie_bins

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_VOCAB = 512
_WINDOW = 24
# Trie and solo runs differ only by bf16 rounding (kernel launch shapes); a leak between samples or a wrong
# position moves the response log-probs by orders of magnitude more.
_LOG_PROB_ATOL = 2e-2
_GRAD_RTOL = 3e-2
# Trie and standard packing round differently, so on one batch either can land a few percent nearer fp64; over the
# eight batches trie's error is 1.01x standard packing's (geometric mean), and 1.12x when a shared row's gradient
# copies are summed in bf16.
_ACCURACY_RATIO = 1.05


def _actor_for(model):
    actor = object.__new__(FSDPTrainRayActor)
    actor.arch_adapter = resolve_arch_adapter(model.config)
    return actor


def _samples(seed: int) -> dict:
    """Two prompts of three samples, a shared system prompt; prompts longer than the window plus flash-attn's
    128-key tile, so window gathers drop ancestors."""
    rng = random.Random(seed)
    system = [rng.randrange(1, _VOCAB) for _ in range(180)]
    tokens, total_lengths, response_lengths = [], [], []
    for _ in range(2):
        prompt = system + [rng.randrange(1, _VOCAB) for _ in range(rng.randrange(10, 30))]
        for _ in range(3):
            response = [rng.randrange(1, _VOCAB) for _ in range(rng.randrange(5, 20))]
            tokens.append(torch.tensor(prompt + response))
            total_lengths.append(len(prompt) + len(response))
            response_lengths.append(len(response))
    return {
        "tokens": tokens,
        # on the GPU, as the actor's rollout data is
        "loss_masks": [torch.ones(r, dtype=torch.int, device="cuda") for r in response_lengths],
        "total_lengths": total_lengths,
        "response_lengths": response_lengths,
    }


_MODELS = pytest.mark.parametrize(
    "kind",
    [
        "full",
        "sliding",
    ],
)


def _tiny_model(kind: str):
    """A tiny model of each ``_MODELS`` kind, in bf16 on the GPU:

    - full, sliding: Qwen3, with sliding-window attention on every other layer for ``sliding``
    """
    torch.manual_seed(0)
    common = dict(
        vocab_size=_VOCAB,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
    )
    sliding = kind == "sliding"
    config = Qwen3Config(
        sliding_window=_WINDOW if sliding else None,
        use_sliding_window=sliding,
        layer_types=["sliding_attention", "full_attention"] * 2 if sliding else ["full_attention"] * 4,
        **common,
    )
    model = Qwen3ForCausalLM._from_config(config, attn_implementation="flash_attention_2", dtype=torch.bfloat16)
    return model.cuda()


@pytest.fixture
def single_rank(monkeypatch):
    state = SimpleNamespace(cp=SimpleNamespace(rank=0, size=1), tp=SimpleNamespace(rank=0, size=1))
    monkeypatch.setattr(data_utils, "get_parallel_state", lambda: state)
    monkeypatch.setattr(context_parallel, "get_parallel_state", lambda: state)


def _response_log_probs(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(logits.float(), dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)


def _solo_log_probs(model, samples) -> list[torch.Tensor]:
    """Each sample alone, as standard (unpacked) training scores it, on the logical 1D positions FSDP passes."""
    model.set_attn_implementation("flash_attention_2")
    out = []
    for index, total in enumerate(samples["total_lengths"]):
        tokens, response = samples["tokens"][index].cuda(), samples["response_lengths"][index]
        positions = torch.arange(total, device="cuda")[None]
        logits = model(
            input_ids=tokens[None],
            position_ids=positions,
            use_cache=False,
        ).logits
        out.append(_response_log_probs(logits[0, total - response - 1 : total - 1], tokens[-response:]))
    return out


def _trie_log_probs(model, samples) -> list[torch.Tensor]:
    """The micro-batch as one trie row, through the actor's input builder and response-only logits."""
    register_trie_attention("flash_attention_2")
    model.set_attn_implementation(trie_attn_implementation(model.config, "flash_attention_2"))
    micro_batch_indices = [list(range(len(samples["tokens"])))]
    model_spec = trie_model_spec(model.config.get_text_config())
    packing_keys = trie_packing_keys(samples)
    trie_micro_batches = TrieMicroBatches(samples, model_spec, micro_batch_indices, packing_keys=packing_keys)
    iterator = data_utils.DataIterator(
        samples, micro_batch_indices=micro_batch_indices, trie_micro_batches=trie_micro_batches
    )
    batch = data_utils.get_batch(iterator, list(samples), pad_multiplier=128, qkv_format="thd")
    layout = batch["trie_layout"]
    if layout.window_kv is not None:
        assert layout.window_kv.index.numel() < layout.kv.index.numel(), "no window gather dropped an ancestor"
    model_args = _actor_for(model)._get_model_inputs_args(batch)
    logits = fsdp_actor._logits(model, model_args, batch)[0]
    targets = torch.cat([t[-r:] for t, r in zip(samples["tokens"], samples["response_lengths"], strict=True)]).cuda()
    return list(_response_log_probs(logits, targets).split(samples["response_lengths"]))


@_MODELS
def test_trie_row_scores_each_sample_as_if_alone(single_rank, kind):
    model = _tiny_model(kind).eval()
    samples = _samples(0)
    with torch.no_grad():
        solo = _solo_log_probs(model, samples)
        trie = _trie_log_probs(model, samples)

    for got, want in zip(trie, solo, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=_LOG_PROB_ATOL)


@_MODELS
def test_fully_sharded_trie_row_scores_each_sample_as_if_alone(nccl_world, single_rank, kind):
    """Under FSDP2, as the actor trains: its input casting hands each wrapped layer a rebuilt copy of the trie row,
    and the actor's check must still see every layer mix over the row."""
    fsdp_args = SimpleNamespace(backend=SimpleNamespace(fp16=False))
    model = fsdp_actor.apply_fsdp2(_tiny_model(kind).eval(), args=fsdp_args)
    samples = _samples(0)
    with torch.no_grad():
        solo = _solo_log_probs(model, samples)
        trie = _trie_log_probs(model, samples)

    for got, want in zip(trie, solo, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=_LOG_PROB_ATOL)


@_MODELS
def test_trie_row_gradients_match_the_solo_sum(single_rank, kind):
    """A shared token's gradient sums over every sample that reads it, once."""
    model = _tiny_model(kind)
    samples = _samples(1)

    model.zero_grad()
    torch.stack([lp.sum() for lp in _solo_log_probs(model, samples)]).sum().backward()
    solo = {name: p.grad.float().clone() for name, p in model.named_parameters()}
    model.zero_grad()
    torch.stack([lp.sum() for lp in _trie_log_probs(model, samples)]).sum().backward()

    for name, p in model.named_parameters():
        error = (p.grad.float() - solo[name]).norm() / solo[name].norm().clamp(min=1e-12)
        bound = _GRAD_RTOL
        assert error <= bound, f"{name}: relative gradient error {error:.3g}"


def _grpo_samples(seed: int) -> tuple[dict, list[float]]:
    """Four prompts of eight samples under a shared system prompt, short responses, and advantages that are
    zero-mean per prompt, as GRPO's are: a shared token's gradient is then a sum that nearly cancels."""
    rng = random.Random(seed)
    system = [rng.randrange(1, _VOCAB) for _ in range(200)]
    tokens, response_lengths, advantages = [], [], []
    for _ in range(4):
        prompt = system + [rng.randrange(1, _VOCAB) for _ in range(300)]
        group = torch.tensor([rng.gauss(0, 1) for _ in range(8)], dtype=torch.float64)
        advantages += ((group - group.mean()) / group.std()).tolist()
        for _ in range(8):
            response = [rng.randrange(1, _VOCAB) for _ in range(rng.randrange(10, 40))]
            tokens.append(torch.tensor(prompt + response))
            response_lengths.append(len(response))
    samples = {
        "tokens": tokens,
        # on the GPU, as the actor's rollout data is
        "loss_masks": [torch.ones(r, dtype=torch.int, device="cuda") for r in response_lengths],
        "total_lengths": [len(t) for t in tokens],
        "response_lengths": response_lengths,
    }
    return samples, advantages


def _grpo_loss(log_probs: list[torch.Tensor], advantages: list[float]) -> torch.Tensor:
    """GRPO's policy loss at ratio 1."""
    return sum(a * lp.double().mean() for a, lp in zip(advantages, log_probs, strict=True)) / len(advantages)


@contextlib.contextmanager
def _plain_pytorch_kernels(model):
    """The model's modeling module on its plain-PyTorch functions, in place of any wrapper bound over them."""
    modeling = sys.modules[type(model).__module__]
    bound = {name: fn for name, fn in vars(modeling).items() if callable(fn) and hasattr(fn, "__wrapped__")}
    for name, fn in bound.items():
        setattr(modeling, name, inspect.unwrap(fn))
    try:
        yield
    finally:
        for name, fn in bound.items():
            setattr(modeling, name, fn)


def _fp64_gradients(model, samples, advantages) -> dict[str, torch.Tensor]:
    """The GRPO loss's gradient in fp64, each sample alone, from the model's weights: on eager attention and the
    modeling module's plain-PyTorch functions.
    """
    reference = type(model)._from_config(model.config, attn_implementation="eager", dtype=torch.float64).cuda()
    reference.load_state_dict({name: value.double() for name, value in model.state_dict().items()})
    log_probs = []
    with _plain_pytorch_kernels(reference):
        for tokens, total, response in zip(
            samples["tokens"], samples["total_lengths"], samples["response_lengths"], strict=True
        ):
            tokens = tokens.cuda()
            logits = reference(input_ids=tokens[None], use_cache=False).logits[0, total - response - 1 : total - 1]
            log_probs.append(torch.log_softmax(logits, dim=-1).gather(-1, tokens[-response:, None]).squeeze(-1))
        _grpo_loss(log_probs, advantages).backward()
    return {name: p.grad for name, p in reference.named_parameters()}


def _packed_log_probs(model, samples) -> list[torch.Tensor]:
    """Standard packing: the micro-batch's samples back to back in one varlen row, response-only logits."""
    model.set_attn_implementation("flash_attention_2")
    iterator = data_utils.DataIterator(samples, micro_batch_indices=[list(range(len(samples["tokens"])))])
    batch = data_utils.get_batch(iterator, list(samples), pad_multiplier=128, qkv_format="thd", get_position_ids=True)
    starts = batch["cu_seqlens_host"][: len(samples["tokens"])]  # the row's padding follows as one more segment
    rows = torch.cat(
        [
            torch.arange(start + total - response - 1, start + total - 1)
            for start, total, response in zip(
                starts, samples["total_lengths"], samples["response_lengths"], strict=True
            )
        ]
    )
    logits = model(**_actor_for(model)._get_model_inputs_args(batch), logits_to_keep=rows.cuda()).logits[0]
    targets = torch.cat([t[-r:] for t, r in zip(samples["tokens"], samples["response_lengths"], strict=True)]).cuda()
    return list(_response_log_probs(logits, targets).split(samples["response_lengths"]))


def _relative_error(grads: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]) -> float:
    difference = sum((grads[name].double() - want).square().sum() for name, want in reference.items())
    return (difference / sum(want.square().sum() for want in reference.values())).sqrt().item()


@_MODELS
def test_trie_row_gradients_are_as_accurate_as_standard_packing(nccl_world, single_rank, kind):
    """With fp32 master weights and bf16 compute, as the actor trains, a trie row's gradient is no further from an
    fp64 reference than standard packing's, over GRPO batches where shared tokens are read by many segments."""
    weights = _tiny_model(kind)
    fsdp_args = SimpleNamespace(backend=SimpleNamespace(fp16=False))
    model = fsdp_actor.apply_fsdp2(_tiny_model(kind).float(), args=fsdp_args)
    ratios = []
    for seed in range(8):
        samples, advantages = _grpo_samples(seed)
        reference = _fp64_gradients(weights, samples, advantages)
        errors = []
        for log_probs in (_packed_log_probs, _trie_log_probs):
            model.zero_grad(set_to_none=True)
            _grpo_loss(log_probs(model, samples), advantages).backward()
            errors.append(
                _relative_error({name: p.grad.full_tensor() for name, p in model.named_parameters()}, reference)
            )
        ratios.append(errors[1] / errors[0])

    assert geometric_mean(ratios) <= _ACCURACY_RATIO, f"trie / standard packing error per batch: {ratios}"


def test_gather_rows_sums_a_shared_rows_copies_in_fp32():
    """A row gathered by many segments gets one fp32 sum of its copies' gradients, rounded to bf16 once, where
    index_select's own backward rounds after every copy (CUDA's atomic bf16 adds) and lands far from the sum."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    source = torch.randn(4, 2, 64, device="cuda", generator=generator).bfloat16()
    rows = torch.tensor([0] * 2000 + [1, 2, 3], device="cuda")
    grad = torch.randn(len(rows), 2, 64, device="cuda", generator=generator).bfloat16()
    exact = torch.zeros(source.shape, dtype=torch.float64, device="cuda").index_add_(0, rows, grad.double())

    assert torch.equal(gather_rows(source, rows), source.index_select(0, rows))
    errors = {}
    for name, gather in (
        ("gather_rows", gather_rows),
        ("index_select", lambda leaf, rows: leaf.index_select(0, rows)),
    ):
        leaf = source.clone().requires_grad_()
        gather(leaf, rows).backward(grad)
        assert leaf.grad.dtype == torch.bfloat16
        errors[name] = ((leaf.grad.double() - exact).norm() / exact.norm()).item()
    # measured: gather_rows about 1.7e-3 (one bf16 rounding), index_select 28-40 times that
    assert errors["gather_rows"] < 2**-8, errors
    assert errors["index_select"] > 10 * errors["gather_rows"], errors


@pytest.mark.parametrize("attn_implementation", sorted(FLASH_ATTN_MODULES))
@pytest.mark.parametrize("window", [None, _WINDOW], ids=["full", "sliding"])
def test_flash_attn_trie_attention_matches_the_reference(window, attn_implementation):
    flash_attn_module = FLASH_ATTN_MODULES[attn_implementation]
    pytest.importorskip(flash_attn_module)
    tokens = [t.tolist() for t in _samples(2)["tokens"]]
    (bin_trie,) = pack_trie_bins(tokens, max_bin_width=sum(len(t) for t in tokens))
    layout = build_trie_layout(
        bin_trie,
        window_left=None if window is None else window - 1,
        window_start_multiple=WINDOW_START_MULTIPLE,
        pad_multiple=128,
    )
    if window is not None:
        assert layout.window_kv.index.numel() < layout.kv.index.numel(), "no window gather dropped an ancestor"
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, heads, layout.num_tokens, 64, device="cuda") for heads in (4, 2, 2))
    trie = TrieAttentionArgs(layout, torch.device("cuda"))

    got, _ = trie_attention_forward(
        None,
        q.bfloat16(),
        k.bfloat16(),
        v.bfloat16(),
        None,
        sliding_window=window,
        flash_attn_module=flash_attn_module,
        miles_trie_attn_args=trie,
    )
    kv = layout.kv if window is None else layout.window_kv
    want = reference_trie_attention(
        *(x[0].transpose(0, 1).double().cpu() for x in (q, k, v)),
        kv,
        layout.cu_seqlens_q,
        window_left=None if window is None else window - 1,
    )
    torch.testing.assert_close(got[0].double().cpu(), want, rtol=0, atol=2e-2)
