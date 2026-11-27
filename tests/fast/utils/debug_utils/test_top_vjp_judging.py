"""The TOP backward gates' judging on CPU: metrics and localized maxima, the contract and quality checks of the VJP
verdict, the cross-layout check, synthetic perturbations each check must fail (one tensor x2 / x0.5, swapped experts,
corrupted CP chunk-boundary rows, a sign flip on a small tensor), capture loading and rollout identity, the GDN
reference path, and DeepSeek-V4's reference shims (deployed KV rounding, per-token mHC terms)."""

import contextlib
import functools
import os
import re
import sys
import types

import pytest
import torch
import transformers

from miles.utils.debug_utils import top_capture, top_layout, top_reference, top_reference_dsv4, top_vjp
from miles.utils.debug_utils.top_compare import compare, summary, worst_experts, worst_row
from miles.utils.debug_utils.top_contract import ContractModule
from miles.utils.debug_utils.top_vjp import (
    _ContractReplay,
    _cotangent_failures,
    _judge_contract,
    _quality_params,
    _quantizes_kv,
    _Replay,
)


def _randn(*shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def _noisy(t, rel, seed=1):
    """``t`` plus noise of relative L2 ``rel``."""
    noise = _randn(*t.shape, seed=seed)
    return t + noise * (rel * t.norm() / noise.norm())


# ---------------------------------------------------------------- metrics


def test_compare_reports_distance_cosine_norm_ratio_and_max_abs():
    b = torch.tensor([3.0, 4.0])
    assert compare(b, b) == dict(rel_l2=0.0, cosine=1.0, norm_ratio=1.0, max_abs=0.0)
    doubled = compare(2 * b, b)
    assert doubled["rel_l2"] == pytest.approx(1.0) and doubled["cosine"] == pytest.approx(1.0)
    assert doubled["norm_ratio"] == pytest.approx(2.0) and doubled["max_abs"] == pytest.approx(4.0)
    assert compare(torch.tensor([-4.0, 3.0]), b)["cosine"] == pytest.approx(0.0)


def test_worst_row_finds_a_few_corrupted_rows_the_whole_tensor_hides():
    reference = _randn(4096, 64)
    top = reference.clone()
    top[2047:2049] *= 1.3  # the two rows at a CP2 chunk boundary
    assert compare(top, reference)["rel_l2"] < 0.01
    assert worst_row(top, reference) == dict(row=2047, rel_l2=pytest.approx(0.3))


def test_worst_row_floors_rows_whose_reference_is_nearly_zero():
    reference = _randn(8, 4)
    reference[3] = 1e-12
    top = reference.clone()
    top[3] = 1e-9  # a rounding-sized difference on a row with no gradient
    assert worst_row(top, reference)["rel_l2"] < 1e-6


def test_worst_experts_reads_stacked_top_and_per_expert_names():
    reference = _randn(16, 8, 4)
    reference[5] *= 0.01
    reference[9] *= 0.01
    swapped = reference.clone()
    swapped[[5, 9]] = swapped[[9, 5]]
    assert compare(swapped, reference)["rel_l2"] < 0.01
    stacked = worst_experts(
        {"m.layers.3.mlp.experts.down_proj": swapped}, {"m.layers.3.mlp.experts.down_proj": reference}
    )
    assert stacked["m.layers.3.mlp.experts"]["expert"] in (5, 9)
    assert stacked["m.layers.3.mlp.experts"]["rel_l2"] > 1
    per_expert = worst_experts(
        {f"m.layers.3.ffn.experts.{e}.w2.weight": swapped[e] for e in range(16)},
        {f"m.layers.3.ffn.experts.{e}.w2.weight": reference[e] for e in range(16)},
    )
    assert per_expert["m.layers.3.ffn.experts"]["expert"] in (5, 9)


def test_summary_keeps_a_zero_gradient_against_a_nonzero_reference():
    reference = torch.tensor([1.0, 2.0])
    comparisons = [compare(torch.zeros(2), reference), compare(torch.zeros(2), torch.zeros(2))]
    result = summary(comparisons)
    assert result["rel_l2_max"] == pytest.approx(1.0) and result["cosine_min"] == 0.0


# ---------------------------------------------------------------- quality check


def _replay(grads, token_sums=None):
    return _Replay(grads=grads, input_grads={}, outputs={}, token_sums=token_sums or {})


def _quality(top, reference, bound, token_sums=None):
    return _quality_params(top, _replay(reference, token_sums), _replay(bound), dp_scale=1.0)


def test_quality_reports_tops_and_the_bound_references_distance():
    reference = _randn(256)
    record = _quality({"p": _noisy(reference, 0.08, seed=2)}, {"p": reference}, {"p": _noisy(reference, 0.6)})["p"]
    assert record["top"]["rel_l2"] == pytest.approx(0.08) and record["bound"]["rel_l2"] == pytest.approx(0.6)


def test_quality_reports_a_near_cancelling_sum_against_its_token_terms():
    reference = torch.tensor([1e-3, 0.0])
    bound = torch.tensor([1.3e-3, 0.0])  # 30% off the net, 0.03% of the terms
    top = torch.tensor([1.5e-3, 0.0])  # 50% off the net, 0.05% of the terms
    params = _quality({"hc": top, "w": top}, {"hc": reference, "w": reference}, {"hc": bound, "w": bound}, {"hc": 1.0})
    assert params["hc"]["cancellation"] == pytest.approx(1e-3)
    assert params["hc"]["judged_vs_token_sum"] == pytest.approx(dict(top=5e-4, bound=3e-4))
    assert "judged_vs_token_sum" not in params["w"]


def _quality_layer(**overrides):
    layer = dict(
        forward=dict(rel_l2=0.004),
        forward_bound=dict(rel_l2=0.003),
        input_grad=dict(rel_l2=0.01),
        input_grad_bound=dict(rel_l2=0.008),
        row=dict(row=7, rel_l2=0.05),
        row_bound=dict(row=3, rel_l2=0.04),
        cotangent_halves_equal=True,
        cotangent_norm=1.0,
    )
    return dict(layer, **overrides)


def test_quality_layers_are_reported_and_only_the_cotangent_is_judged():
    assert _cotangent_failures({2: _quality_layer(forward=dict(rel_l2=0.3), row=dict(row=511, rel_l2=0.9))}) == []
    assert _cotangent_failures({2: _quality_layer(cotangent_norm=0.0)}) == [
        "layer 2: zero output cotangent, no gradient signal to compare"
    ]
    assert _cotangent_failures({2: _quality_layer(cotangent_halves_equal=False)}) == [
        "layer 2: the halves of the packed output cotangent differ"
    ]


# ---------------------------------------------------------------- contract check


def _contract(reference_grad, reference_params, *, visible=None, module="mlp"):
    """A contract replay of one captured module (layer 3) whose surrogate reproduced ``visible``."""
    visible = reference_grad if visible is None else visible
    record = dict(reference=module, forward=[compare(visible, reference_grad)])
    return _ContractReplay(
        modules={(3, module): record}, grads=reference_params, input_grads={(3, module): [reference_grad]}
    )


def _contract_case():
    reference_grad = _randn(4096, 32)
    experts = _randn(16, 8, 4, seed=3)
    experts[5] *= 0.01
    experts[9] *= 0.01
    params = {
        "model.layers.3.mlp.experts.down_proj": experts,
        "model.layers.3.mlp.gate.weight": _randn(16, 32, seed=4),
        "model.layers.3.mlp.shared_expert_gate.weight": _randn(1, 4, seed=5),
    }
    return reference_grad, params


def _judge(top_grad, top_params, reference_grad, params, **kwargs):
    top = dict(top_params, **{"model.layers.3.input_layernorm.weight": torch.ones(32)})
    captures = [{3: {"mlp.input_grad": top_grad[:, None]}}]
    return _judge_contract("Qwen GDN MoE", top, captures, _contract(reference_grad, params, **kwargs), dp_scale=1.0)


def test_contract_passes_backward_rounding():
    reference_grad, params = _contract_case()
    # Rounding scales with each element, so a small expert's gradient keeps a small relative error.
    top_params = {name: grad * (1 + 1e-3 * _randn(*grad.shape, seed=7)) for name, grad in params.items()}
    record, failures = _judge(_noisy(reference_grad, 1e-3), top_params, reference_grad, params)
    assert failures == []
    assert record["uncovered"] == ["model.layers.3.input_layernorm.weight"]


@pytest.mark.parametrize("scale", [2.0, 0.5])
def test_contract_fails_one_tensor_scaled(scale):
    reference_grad, params = _contract_case()
    top_params = dict(params, **{"model.layers.3.mlp.gate.weight": params["model.layers.3.mlp.gate.weight"] * scale})
    _, failures = _judge(reference_grad, top_params, reference_grad, params)
    assert failures == ["model.layers.3.mlp.gate.weight (contract)"]


def test_contract_fails_swapped_experts():
    reference_grad, params = _contract_case()
    swapped = params["model.layers.3.mlp.experts.down_proj"].clone()
    swapped[[5, 9]] = swapped[[9, 5]]
    top_params = dict(params, **{"model.layers.3.mlp.experts.down_proj": swapped})
    _, failures = _judge(reference_grad, top_params, reference_grad, params)
    assert len(failures) == 1 and failures[0].startswith("model.layers.3.mlp.experts: expert")


def test_contract_fails_corrupted_cp_chunk_boundary_rows():
    reference_grad, params = _contract_case()
    top_grad = reference_grad.clone()
    top_grad[2047:2049] *= 1.3
    _, failures = _judge(top_grad, params, reference_grad, params)
    assert failures == ["layer 3 mlp input grad"]


def test_contract_fails_a_sign_flip_on_a_small_tensor():
    reference_grad, params = _contract_case()
    name = "model.layers.3.mlp.shared_expert_gate.weight"
    _, failures = _judge(reference_grad, dict(params, **{name: -params[name]}), reference_grad, params)
    assert failures == [f"{name} (contract)"]


def test_contract_refuses_a_surrogate_that_misses_the_visible_output():
    reference_grad, params = _contract_case()
    _, failures = _judge(reference_grad, params, reference_grad, params, visible=_noisy(reference_grad, 0.05))
    assert len(failures) == 1 and "fix the reference" in failures[0]


def _visible_gpu(scale):
    return dict(gpu=os.environ["CUDA_VISIBLE_DEVICES"], value=scale * torch.ones(2))


def _broken():
    raise ValueError("no")


def test_replays_run_one_process_per_gpu(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    jobs = dict(
        contract=functools.partial(_visible_gpu, 1.0), exact=functools.partial(_visible_gpu, 2.0), masters=None
    )
    results = top_vjp._run_replays(jobs, scratch=tmp_path)
    assert results["masters"] is None
    assert {results[name]["gpu"] for name in ("contract", "exact")} == {"3", "5"}
    assert torch.equal(results["exact"]["value"], torch.full((2,), 2.0))
    with pytest.raises(RuntimeError, match="the exact replay failed"):
        top_vjp._run_replays(dict(contract=functools.partial(_visible_gpu, 1.0), exact=_broken), scratch=tmp_path)
    # Fewer GPUs than replays: they run here, in turn.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert top_vjp._run_replays(jobs, scratch=tmp_path)["exact"]["gpu"] == "0"


def test_contract_fails_a_family_without_measured_floors():
    reference_grad, params = _contract_case()
    captures = [{3: {"mlp.input_grad": reference_grad[:, None]}}]
    _, failures = _judge_contract("Qwen3 MoE", params, captures, _contract(reference_grad, params), dp_scale=1.0)
    assert failures and all("no measured contract floor for Qwen3 MoE mlp" in failure for failure in failures)


def test_contract_judges_a_near_cancelling_parameter_against_its_token_terms(monkeypatch):
    """An mHC module: its input gradient also carries the residual path, so only its parameters are judged."""
    monkeypatch.setitem(top_vjp.CONTRACT_TOKEN_SUM_FLOORS, ("Qwen GDN MoE", "mlp"), 1e-4)
    reference_grad, params = _contract_case()
    name = "model.layers.3.mlp.shared_expert_gate.weight"
    token_sum = 1e3 * float(params[name].norm())
    contract = _contract(reference_grad, params)._replace(token_sums={name: token_sum})
    contract.modules[3, "mlp"]["judges_input_grad"] = False
    captures = [{3: {"mlp.input_grad": 2 * reference_grad[:, None]}}]

    def judge(grad):
        return _judge_contract("Qwen GDN MoE", dict(params, **{name: grad}), captures, contract, dp_scale=1.0)

    record, failures = judge(params[name] * 1.05)
    assert failures == [] and record["params"][name]["cancellation"] == pytest.approx(1e-3)
    assert judge(params[name] * 1.5)[1] == [f"{name} (contract)"]
    assert judge(-params[name])[1] == [f"{name} (contract)"]


def test_contract_replays_every_output_of_a_module(monkeypatch):
    """A module with two outputs (as DeepSeek-V4's mHC): the bridge takes each output's own cotangent."""
    monkeypatch.setattr(top_vjp, "DEVICE", "cpu")
    layer = torch.nn.Linear(4, 3, bias=False)
    spec = ContractModule(
        params=("weight",),
        run=lambda model, layer, x, positions, causal: (layer(x), x.square().sum(-1, keepdim=True)),
        outputs=("output.0", "output.1"),
    )
    adapter = top_reference.FamilyAdapter(
        layers=lambda model: [layer],
        run_layer=None,
        to_hf_param_grads=None,
        load=lambda path, dtype: torch.nn.Module(),
        replay=lambda *args: contextlib.nullcontext(),
        contract=lambda layer: {"m": spec},
    )
    x = _randn(5, 4).float()
    dy0, dy1 = _randn(5, 3, seed=2).float(), _randn(5, 1, seed=3).float()
    slot = {"m.input": x, "m.output.0": layer(x).detach(), "m.output.1": x.square().sum(-1, keepdim=True)}
    slot |= {"m.output.0_grad": dy0, "m.output.1_grad": dy1}
    capture = {"cu_seqlens": torch.tensor([0, 2, 5]), 0: slot}
    replay = top_vjp._replay_contract(adapter, "unused", captures=[capture], layers=[0], quantize_kv=False)
    assert torch.allclose(replay.grads["model.layers.0.weight"], dy0.T @ x, atol=1e-6)
    assert torch.allclose(replay.input_grads[0, "m"][0], dy0 @ layer.weight.detach() + 2 * x * dy1, atol=1e-6)
    assert replay.modules[0, "m"]["forward"][0]["rel_l2"] < 1e-6


def test_contract_hands_carried_state_only_to_a_module_that_takes_it(monkeypatch):
    """Qwen3.8-Flash-Next's PLE layer: its attention hyper-connection hashes the document's own token ids, while the
    other modules of the same layer keep the plain run signature."""
    monkeypatch.setattr(top_vjp, "DEVICE", "cpu")
    layer = torch.nn.Linear(4, 4, bias=False)

    def shifted(model, layer, x, positions, causal, *, carried):
        return (layer(x) + carried.replayed["input_ids"][None, :, None].to(x),)

    adapter = top_reference.FamilyAdapter(
        layers=lambda model: [layer],
        run_layer=None,
        to_hf_param_grads=None,
        load=lambda path, dtype: torch.nn.Module(),
        replay=lambda *args: contextlib.nullcontext(),
        contract=lambda layer: {
            "plain": ContractModule(params=("weight",), run=lambda model, layer, x, positions, causal: (layer(x),)),
            "shifted": ContractModule(params=("weight",), run=shifted, takes_carried=True),
        },
        carried=lambda model, capture, number, lo, hi: top_reference.Carried(
            inputs={}, input_grads={}, cotangents={}, replayed={"input_ids": capture["input_ids"][lo:hi]}
        ),
    )
    x, ids = _randn(5, 4).float(), torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0])
    out = layer(x).detach()
    slot = {"plain.input": x, "plain.output": out, "plain.output_grad": torch.ones(5, 4)}
    slot |= {"shifted.input": x, "shifted.output": out + ids[:, None], "shifted.output_grad": torch.ones(5, 4)}
    capture = {"cu_seqlens": torch.tensor([0, 2, 5]), "input_ids": ids, 0: slot}
    replay = top_vjp._replay_contract(adapter, "unused", captures=[capture], layers=[0], quantize_kv=False)
    assert replay.modules[0, "plain"]["forward"][0]["rel_l2"] < 1e-6
    assert replay.modules[0, "shifted"]["forward"][0]["rel_l2"] < 1e-6


def test_layout_judges_a_near_cancelling_gradient_against_its_token_terms():
    """DeepSeek-V4's mHC biases: reduction order moves a near-cancelling net by multiples of its own size."""
    base, phase = _layout_phase(), _layout_phase()
    name, bias = "module.module.decoder.layers.3.bias", _randn(24, seed=9) * 1e-6
    base["params"][name], phase["params"][name] = bias, bias * 3
    token_sums = {"decoder.layers.3.bias": 1.0}
    assert top_layout._compare_phase(phase, base, [3], token_sums)["failures"] == []
    assert top_layout._compare_phase(phase, base, [3], {})["failures"] == [name]
    phase["params"][name] = -bias
    assert top_layout._compare_phase(phase, base, [3], token_sums)["failures"] == [name]


def test_layout_judges_a_scalar_gradient_by_its_sign():
    """DeepSeek-V4's mHC alphas: near-cancelling sums whose net moves by multiples with the reduction order."""
    base, phase = _layout_phase(), _layout_phase()
    base["params"]["decoder.layers.3.alpha_post"] = torch.tensor([1e-3])
    phase["params"]["decoder.layers.3.alpha_post"] = torch.tensor([3.3e-3])
    assert top_layout._compare_phase(phase, base, [3], {})["failures"] == []
    phase["params"]["decoder.layers.3.alpha_post"] = torch.tensor([-1e-3])
    assert top_layout._compare_phase(phase, base, [3], {})["failures"] == ["decoder.layers.3.alpha_post"]


def test_visible_with_surrogate_takes_the_value_of_one_and_the_gradient_of_the_other():
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    visible = torch.tensor([10.0, 20.0])
    out = top_reference.VisibleWithSurrogate.apply(visible, 3 * x)
    assert torch.equal(out, visible)
    out.backward(torch.tensor([1.0, -1.0]))
    assert torch.equal(x.grad, torch.tensor([3.0, -3.0]))
    with pytest.raises(ValueError, match="differ"):
        top_reference.VisibleWithSurrogate.apply(visible.double(), 3 * x)


# ---------------------------------------------------------------- references


def test_torch_gdn_path_swaps_installed_kernels_for_the_torch_implementations():
    module = types.ModuleType("fake_modeling_gdn")

    def torch_chunk_gated_delta_rule(q):
        return "torch"

    @functools.wraps(torch_chunk_gated_delta_rule)
    def wrapped(q):
        return "fla"

    module.torch_chunk_gated_delta_rule = wrapped
    layer_cls = type("FakeGatedDeltaNet", (torch.nn.Module,), {"__module__": module.__name__})
    sys.modules[module.__name__] = module
    try:
        unregistered = type("VendorBlock", (torch.nn.Module,), {"__module__": "not_in_sys_modules"})
        layer = torch.nn.Sequential(layer_cls(), unregistered())
        with top_reference.torch_gdn_path(layer):
            assert module.torch_chunk_gated_delta_rule(None) == "torch"
        assert module.torch_chunk_gated_delta_rule(None) == "fla"
    finally:
        del sys.modules[module.__name__]


def test_fp8_kv_needs_a_family_reference_that_rounds_it():
    adapter = top_vjp.ADAPTERS["Qwen3 dense"]
    assert _quantizes_kv(adapter, "Qwen3 dense", None) is False
    assert _quantizes_kv(adapter, "Qwen3 dense", "bfloat16") is False
    with pytest.raises(ValueError, match="cannot round its KV cache"):
        _quantizes_kv(adapter, "Qwen3 dense", "fp8_e4m3")
    assert _quantizes_kv(top_vjp.ADAPTERS["DeepSeek-V4"], "DeepSeek-V4", "fp8_e4m3") is True
    assert _quantizes_kv(top_vjp.ADAPTERS["GLM"], "GLM", "fp8_e4m3") is True
    # Its reference rounds as E4M3; another FP8 cache would be judged against the wrong rounding.
    with pytest.raises(ValueError, match="E4M3 only"):
        _quantizes_kv(top_vjp.ADAPTERS["GLM"], "GLM", "fp8_e5m2")


def test_every_gated_family_declares_contract_modules():
    gated = {"Qwen3 dense", "Qwen GDN MoE", "Qwen3.8-Flash-Next", "GLM", "GLM-5.3-Flash", "DeepSeek-V4"}
    # DeepSeek-V4.1's reference declares no contract modules yet: its verdict reports the replays only.
    assert set(top_vjp.ADAPTERS) == gated | {"DeepSeek-V4.1"}
    assert all(top_vjp.ADAPTERS[family].contract is not None for family in gated)
    assert top_vjp.ADAPTERS["DeepSeek-V4.1"].contract is None


def _tiny_qwen3(path):
    config = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    transformers.Qwen3ForCausalLM(config).save_pretrained(path)
    return str(path)


def _capture_hf_as_top(model_path, dump_dir, number, lengths, scale_param=None):
    """A capture of decoder layer ``number`` whose "TOP" is HF itself in FP32, in the dumper's format: layer and module
    inputs, outputs and gradients per packed microbatch, and the parameter gradients (``scale_param`` scaled)."""
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32, attn_implementation="eager"
    )
    layer, hidden = model.model.layers[number], model.config.hidden_size
    rows, seen = {}, {}

    def pre(name):
        def hook(module, args, kwargs):
            seen[f"{name}.inputs.0"] = value = args[0] if args else kwargs["hidden_states"]
            value.retain_grad()

        return hook

    def post(name):
        def hook(module, args, output):
            seen[f"{name}.output"] = value = output[0] if isinstance(output, tuple) else output
            value.retain_grad()

        return hook

    for name in ("self_attn", "mlp"):
        layer.get_submodule(name).register_forward_pre_hook(pre(name), with_kwargs=True)
        layer.get_submodule(name).register_forward_hook(post(name))
    generator = torch.Generator().manual_seed(1)
    for n in lengths:
        x = torch.randn(1, n, hidden, generator=generator).requires_grad_()
        positions, causal = torch.arange(n)[None], torch.full((n, n), torch.finfo(torch.float32).min).triu_(1)
        out = top_reference._qwen3_run_layer(model, layer, x, positions, causal[None, None])
        cotangent = torch.randn(out.shape, generator=generator)
        out.backward(cotangent)
        seen["inputs.hidden_states"] = x
        for key, value in seen.items():
            rows.setdefault(key, []).append((value.detach()[0], value.grad[0]))
        rows.setdefault("output", []).append((out.detach()[0], cotangent[0]))
    top_name = {"self_attn": "self_attention", "mlp": "mlp"}
    directory = dump_dir / "fwd_bwd" / "rollout_0"
    directory.mkdir(parents=True)
    index = 0

    def dump(name, value):
        nonlocal index
        index += 1
        meta = dict(rank=0, name=name, step=0, megatron_parallel_info={})
        torch.save(dict(meta=meta, value=value), directory / f"rank=0___dump_index={index}.pt")

    dump("cu_seqlens_q", torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()]))
    for key, pairs in rows.items():
        module, _, kind = key.partition(".") if key.split(".")[0] in top_name else ("", "", key)
        name = f"non_intrusive__module.decoder.layers.{number}.{f'{top_name[module]}.' if module else ''}{kind}"
        value, grad = (torch.cat(column)[:, None] for column in zip(*pairs, strict=True))
        dump(name, value)
        dump(f"grad__{name}", grad)
    for name, param in layer.named_parameters():
        grad = param.grad * (2.0 if name == scale_param else 1.0)
        dump(f"grad__param__module.decoder.layers.{number}.{name}", grad)


@pytest.fixture
def tiny_family(monkeypatch):
    adapter = top_reference.FamilyAdapter(
        layers=lambda model: model.model.layers,
        run_layer=top_reference._qwen3_run_layer,
        to_hf_param_grads=lambda grads, config: {
            re.sub(r"^.*decoder\.layers", "model.layers", name): grad for name, grad in grads.items()
        },
        contract=top_reference._qwen3_contract,
    )
    monkeypatch.setitem(top_vjp.ADAPTERS, "tiny", adapter)
    for floors in (top_vjp.CONTRACT_FLOORS, top_vjp.CONTRACT_LOCAL_FLOORS, top_vjp.CONTRACT_FORWARD_FLOORS):
        for module in ("self_attention", "mlp"):
            monkeypatch.setitem(floors, ("tiny", module), 1e-4)
    monkeypatch.setattr(top_vjp, "DEVICE", "cpu")
    return "tiny"


@pytest.mark.parametrize("scale_param", [None, "self_attn.o_proj.weight"])
def test_verdict_end_to_end_on_cpu(tmp_path, tiny_family, scale_param):
    """HF in FP32 as "TOP" passes its contract (FP32 at the same inputs) and sits at FP32 in the reported quality; the
    same capture with one parameter gradient doubled fails the contract, and the quality reports it 100% off."""
    model_path = _tiny_qwen3(tmp_path / "model")
    _capture_hf_as_top(model_path, tmp_path / "dump", 1, [7, 12], scale_param=scale_param)
    judge = dict(family=tiny_family, model_path=model_path, dump_dir=str(tmp_path / "dump"), layers=[1])
    verdict = top_vjp.vjp_verdict(**judge, replay_path=str(tmp_path / "replays.pt"))
    # The saved replays judge the capture again without replaying (the meta-gate's path).
    assert top_vjp.vjp_verdict(**judge, replay_path=str(tmp_path / "replays.pt"))["failures"] == verdict["failures"]
    contract = verdict["contract"]
    assert set(contract["modules"]) == {"1.self_attention", "1.mlp"}
    assert contract["uncovered"] == [
        "model.layers.1.input_layernorm.weight",
        "model.layers.1.post_attention_layernorm.weight",
    ]
    if scale_param is None:
        assert verdict["failures"] == []
        assert contract["summary"]["rel_l2_max"] < 1e-5
        assert verdict["quality"]["summary"]["rel_l2_max"] < 1e-5
    else:
        name = f"model.layers.1.{scale_param}"
        assert verdict["failures"] == [f"{name} (contract)"]
        assert verdict["quality"]["params"][name]["top"]["rel_l2"] == pytest.approx(1.0)


class _Gain(torch.nn.Module):
    """A norm as Qwen3's q/k norms: its weight times the RMS-normalized input."""

    def __init__(self, dim, seed):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(dim, generator=torch.Generator().manual_seed(seed)) + 0.5)

    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6))


def test_qwen3_token_terms_sum_each_tokens_share():
    layer = torch.nn.Module()
    layer.self_attn = torch.nn.Module()
    layer.self_attn.q_norm, layer.self_attn.k_norm = _Gain(8, seed=1), _Gain(8, seed=2)
    g = torch.Generator().manual_seed(3)
    x, cotangent = torch.randn(1, 5, 2, 8, generator=g), torch.randn(1, 5, 2, 8, generator=g)
    weight = layer.self_attn.q_norm.weight

    def loss(rows=None):
        rows = slice(None) if rows is None else rows
        return (layer.self_attn.q_norm(x[:, rows]) * cotangent[:, rows]).sum()

    plain = torch.autograd.grad(loss(), weight)[0]
    with top_reference._qwen3_token_terms(layer) as sums:
        value = loss()
        value.backward()
    assert torch.equal(value, loss())
    assert torch.allclose(weight.grad, plain, rtol=1e-6, atol=1e-7)
    # Each token's share is the gradient of its own rows, summed over its heads; the unused key norm records none.
    per_token = [float(torch.autograd.grad(loss(slice(t, t + 1)), weight)[0].double().norm()) for t in range(5)]
    assert sums == {"self_attn.q_norm.weight": pytest.approx(sum(per_token), rel=1e-5)}


# ---------------------------------------------------------------- cross-layout check


def _sequence(seed, *, n=64, hidden=8):
    return {
        3: {
            "input_grad": _randn(n, 1, 2 * hidden, seed=seed),
            "output_grad": _randn(n, 1, 2 * hidden, seed=seed + 100),
            "routing": torch.zeros(n, 16, dtype=torch.bool).index_fill_(1, torch.tensor([seed % 16]), True),
        }
    }


def _layout_phase(scale=1.0, *, seeds=(0, 1, 2)):
    experts = {f"decoder.layers.3.mlp.experts.linear_fc2.weight{e}": _randn(4, 4, seed=50 + e) for e in range(8)}
    experts["decoder.layers.3.mlp.experts.linear_fc2.weight5"] *= 0.01
    experts["decoder.layers.3.mlp.experts.linear_fc2.weight6"] *= 0.01
    params = dict(experts, **{"decoder.layers.3.input_layernorm.weight": _randn(8, seed=60)})
    sequences = {i: _sequence(seed) for i, seed in enumerate(seeds)}
    for sequence in sequences.values():
        for kind in ("input_grad", "output_grad"):
            sequence[3][kind] = sequence[3][kind] * scale
    return dict(params=params, sequences=sequences)


def test_layout_fits_the_phase_loss_normalization_and_passes_reduction_order():
    base, phase = _layout_phase(), _layout_phase(scale=4.0)
    phase["params"] = {name: _noisy(grad, 1e-4) for name, grad in phase["params"].items()}
    record = top_layout._compare_phase(phase, base, [3], {})
    assert record["failures"] == [] and record["activation_grad_scale"] == pytest.approx(0.25)


def test_layout_fails_corrupted_cp_chunk_boundary_rows():
    base, phase = _layout_phase(), _layout_phase()
    phase["sequences"][1][3]["input_grad"][31:33] *= 1.3
    assert top_layout._compare_phase(phase, base, [3], {})["failures"] == ["layer 3 input_grad"]


@pytest.mark.parametrize("scale", [2.0, 0.5])
def test_layout_fails_one_parameter_scaled(scale):
    base, phase = _layout_phase(), _layout_phase()
    phase["params"]["decoder.layers.3.input_layernorm.weight"] *= scale
    assert top_layout._compare_phase(phase, base, [3], {})["failures"] == ["decoder.layers.3.input_layernorm.weight"]


def test_layout_fails_swapped_experts():
    base, phase = _layout_phase(), _layout_phase()
    name = "decoder.layers.3.mlp.experts.linear_fc2.weight{}"
    phase["params"][name.format(5)], phase["params"][name.format(6)] = (
        base["params"][name.format(6)],
        base["params"][name.format(5)],
    )
    failures = top_layout._compare_phase(phase, base, [3], {})["failures"]
    # Each of TOP's experts is its own parameter, so the two also fail one by one.
    assert failures[:2] == [name.format(5), name.format(6)]
    assert failures[2].startswith("decoder.layers.3.mlp.experts: expert")


def test_layout_fails_a_sign_flip_on_a_small_tensor_and_a_different_routing():
    base, phase = _layout_phase(), _layout_phase()
    phase["params"]["decoder.layers.3.input_layernorm.weight"][:2] *= -1
    phase["sequences"][2][3]["routing"] = phase["sequences"][2][3]["routing"].roll(1, dims=1)
    failures = top_layout._compare_phase(phase, base, [3], {})["failures"]
    assert failures == ["layer 3 routing differs on samples [2]", "decoder.layers.3.input_layernorm.weight"]


def _views(samples, sequences, cp):
    """A DP1 capture's views at ``cp``: every sample's packed stream and a padding segment, cut into CP slices."""
    tokens = torch.cat([torch.tensor(s["tokens"]) for s in samples] + [torch.zeros(8, dtype=torch.long)])
    bounds = [0]
    for s in samples:
        bounds.append(bounds[-1] + len(s["tokens"]))
    bounds.append(len(tokens))
    rows = {
        kind: torch.cat([sequences[i][3][kind] for i in range(len(samples))] + [_randn(8, 1, 16)])
        for kind in ("input_grad", "output_grad")
    }
    rows["routing"] = torch.cat(
        [sequences[i][3]["routing"] for i in range(len(samples))] + [torch.zeros(8, 16, dtype=torch.bool)]
    )
    local = len(tokens) // cp
    return {
        (0, rank, 0): {
            "cu_seqlens": torch.tensor(bounds),
            "input_ids": tokens[rank * local : (rank + 1) * local],
            3: {kind: value[rank * local : (rank + 1) * local] for kind, value in rows.items()},
        }
        for rank in range(cp)
    }


def test_layout_finds_each_sequence_across_cp_slices():
    samples = [dict(index=i, tokens=list(range(10 * i + 1, 10 * i + 65))) for i in range(3)]
    expected = {i: _sequence(seed) for i, seed in enumerate((0, 1, 2))}
    for cp in (1, 4):
        found = top_layout._sequences(_views(samples, expected, cp), samples, [3])
        assert set(found) == {0, 1, 2}
        for i in found:
            for kind, rows in expected[i][3].items():
                assert torch.equal(found[i][3][kind], rows)


def test_layout_finds_each_sequence_across_gdn_cp_slices():
    """GDN TOP's all-gather CP: each rank's bounds are local, and a sequence cut by a rank boundary spans two."""
    samples = [dict(index=i, tokens=list(range(10 * i + 1, 10 * i + 65))) for i in range(3)]
    expected = {i: _sequence(seed) for i, seed in enumerate((0, 1, 2))}
    views = _views(samples, expected, 4)
    for rank in range(4):
        view = views[0, rank, 0]
        n = len(view["input_ids"])
        start = rank * n
        global_bounds = [0, 64, 128, 192, 200]
        local = sorted({0, n} | {b - start for b in global_bounds if start < b < start + n})
        view["cu_seqlens"] = torch.tensor(local)
    found = top_layout._sequences(views, samples, [3])
    assert set(found) == {0, 1, 2}
    for i in found:
        for kind, rows in expected[i][3].items():
            assert torch.equal(found[i][3][kind], rows)


# ---------------------------------------------------------------- capture


def _sample(index, tokens, reward=0.5):
    return dict(index=index, tokens=tokens, response_length=2, reward=reward)


def _save_rollout(path, samples):
    (path / top_capture.ROLLOUT_DATA).mkdir(parents=True)
    torch.save(dict(rollout_id=0, samples=samples), path / top_capture.ROLLOUT_DATA / "0.pt")
    return str(path)


def test_phases_must_train_on_the_same_rollouts(tmp_path):
    a = _save_rollout(tmp_path / "cp1", [_sample(0, [1, 2, 3]), _sample(1, [1, 2, 4])])
    b = _save_rollout(tmp_path / "cp4", [_sample(1, [1, 2, 4]), _sample(0, [1, 2, 3])])
    c = _save_rollout(tmp_path / "pp2", [_sample(0, [1, 2, 3]), _sample(1, [1, 2, 5])])
    top_capture.check_same_rollouts({"cp1": a, "cp4": b})
    with pytest.raises(AssertionError, match="phases trained on different rollouts"):
        top_capture.check_same_rollouts({"cp1": a, "pp2": c})


def test_capture_args_name_each_stage_s_local_layers():
    args = top_capture.capture_args(dump_dir="/d", layers=(2, 3), stage_layers=(3, 1))
    assert "decoder.layers.(0|2).(inputs.hidden_states|output)$" in args
    assert 'search("^param__.*decoder.layers.(2|3).", name)' in args
    assert "--save-debug-rollout-data /d/rollout_data/{rollout_id}.pt" in args
    args = top_capture.capture_args(dump_dir="/d", layers=(2, 3))
    assert "decoder.layers.(2|3).(self_attention|mlp).(inputs.0|output)$" in args
    hc = "self_attention_hyper_connection|mlp_hyper_connection"
    assert f"decoder.layers.(2|3).({hc}).(inputs.0|output.[012])$" in args


def _dump(directory, index, *, rank, name, value, pp_rank=0, pp_size=1, recompute=None):
    meta = dict(rank=rank, name=name, step=0, megatron_parallel_info=dict(pp_rank=pp_rank, pp_size=pp_size))
    if recompute:
        meta["recompute_status"] = recompute
    torch.save(dict(meta=meta, value=value), directory / f"step=0___rank={rank}___dump_index={index}.pt")


def test_load_captures_renumbers_uneven_pipeline_stages(tmp_path):
    directory = tmp_path / "fwd_bwd" / "rollout_0"
    directory.mkdir(parents=True)
    prefix = "non_intrusive__module.module.decoder.layers"
    _dump(directory, 1, rank=0, name=f"{prefix}.2.output", value=torch.ones(2))
    _dump(directory, 2, rank=0, name=f"{prefix}.0.output", value=torch.zeros(2))
    _dump(directory, 3, rank=1, name=f"{prefix}.0.output", value=torch.full((2,), 3.0), pp_rank=1, pp_size=2)
    _dump(directory, 4, rank=1, name=f"{prefix}.0.mlp.inputs.0", value=torch.full((2,), 4.0), pp_rank=1, pp_size=2)
    _dump(directory, 5, rank=1, name=f"grad__{prefix}.0.mlp.output", value=torch.full((2,), 5.0), pp_rank=1, pp_size=2)
    _dump(
        directory,
        6,
        rank=1,
        name=f"{prefix}.0.mlp.inputs.0",
        value=torch.full((2,), 6.0),
        pp_rank=1,
        pp_size=2,
        recompute="recompute",
    )
    _dump(directory, 7, rank=0, name="input_ids", value=torch.tensor([[7, 8]]))
    hc = f"{prefix}.2.mlp_hyper_connection"
    _dump(directory, 8, rank=0, name=f"{hc}.output.1", value=torch.full((2,), 8.0))
    _dump(directory, 9, rank=0, name=f"grad__{hc}.output.1", value=torch.full((2,), 9.0))
    for pp_size in (2,):
        _dump(directory, 0, rank=0, name="cu_seqlens_q", value=torch.tensor([0, 2]), pp_size=pp_size)
    ranks = top_capture.load_captures(str(tmp_path), num_layers=4, layers=(2, 3), stage_layers=(3, 1))
    assert set(ranks[0].steps[0]) == {"cu_seqlens", "input_ids", 2}
    assert torch.equal(ranks[1].steps[0][3]["output"], torch.full((2,), 3.0))
    assert torch.equal(ranks[1].steps[0][3]["mlp.input_recompute"], torch.full((2,), 6.0))
    assert torch.equal(ranks[1].steps[0][3]["mlp.output_grad"], torch.full((2,), 5.0))
    assert torch.equal(ranks[0].steps[0][2]["mlp_hyper_connection.output.1_grad"], torch.full((2,), 9.0))
    with pytest.raises(ValueError, match="stage_layers"):
        top_capture.load_captures(str(tmp_path), num_layers=4, layers=(2, 3))


# ---------------------------------------------------------------- DeepSeek-V4 reference shims


def test_dsv4_kv_rounding_is_in_place_and_straight_through():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 5, 512, generator=g)
    leaf = x.clone().requires_grad_()
    kv = leaf * 1.0
    with top_reference_dsv4.kv_quantization(None):
        top_reference_dsv4._act_quant(kv[..., :-64], 64, None, "fp32", True)
    expected = torch.cat([top_reference_dsv4.fp8_group_fake_quant(x[..., :-64], 64), x[..., -64:]], -1)
    assert torch.equal(kv, expected) and not torch.equal(kv, x)
    kv.backward(torch.full_like(kv, 3.0))
    assert torch.equal(leaf.grad, torch.full_like(x, 3.0))
    # Outside kv_quantization the rows stay unrounded (the unquantized reference).
    untouched = x.clone()
    top_reference_dsv4._act_quant(untouched[..., :-64], 64, None, "fp32", True)
    assert torch.equal(untouched, x)
    with pytest.raises(NotImplementedError):
        top_reference_dsv4._act_quant(x, 128, None, "fp32")


class _Hc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(1)
        self.hc_attn_base = torch.nn.Parameter(torch.randn(24, generator=g) * 0.1)
        self.hc_attn_scale = torch.nn.Parameter(torch.rand(3, generator=g) + 0.5)
        self.other = torch.nn.Parameter(torch.zeros(3))


def test_dsv4_token_terms_sum_each_tokens_share():
    layer = _Hc()
    g = torch.Generator().manual_seed(2)
    mixes = torch.randn(1, 6, 24, generator=g)
    weights = [torch.randn(1, 6, *s, generator=g) for s in ((4,), (4,), (4, 4))]

    def loss(m):
        outs = top_reference_dsv4._hc_split_sinkhorn(m, layer.hc_attn_scale, layer.hc_attn_base)
        return sum((o * w[:, : o.shape[1]]).sum() for o, w in zip(outs, weights, strict=True))

    plain = torch.autograd.grad(loss(mixes), [layer.hc_attn_base, layer.hc_attn_scale])
    with top_reference_dsv4.token_terms(layer) as sums:
        recorded = torch.autograd.grad(loss(mixes), [layer.hc_attn_base, layer.hc_attn_scale], retain_graph=True)
        loss(mixes).backward()
    assert set(sums) == {"hc_attn_base", "hc_attn_scale"}
    for a, b in zip(plain, recorded, strict=True):
        assert torch.allclose(a, b, rtol=1e-6, atol=1e-7)
    # The terms are independent per token: each token's share is the gradient of its own rows.
    for name, param in (("hc_attn_base", layer.hc_attn_base), ("hc_attn_scale", layer.hc_attn_scale)):
        per_token = []
        for t in range(6):
            outs = top_reference_dsv4._hc_split_sinkhorn(mixes[:, t : t + 1], layer.hc_attn_scale, layer.hc_attn_base)
            part = sum((o * w[:, t : t + 1]).sum() for o, w in zip(outs, weights, strict=True))
            per_token.append(float(torch.autograd.grad(part, param)[0].double().norm()))
        assert sums[name] == pytest.approx(sum(per_token), rel=1e-5)


def test_dsv4_token_terms_cover_the_attention_sink():
    layer = torch.nn.Module()
    layer.attn = torch.nn.Module()
    layer.attn.attn_sink = torch.nn.Parameter(torch.tensor([0.3, -0.2]))
    g = torch.Generator().manual_seed(4)
    q, kv = torch.randn(1, 5, 2, 8, generator=g), torch.randn(1, 6, 8, generator=g)
    topk = torch.randint(0, 6, (1, 5, 3), generator=g)
    weight = torch.randn(1, 5, 2, 8, generator=g)

    def loss(rows=None):
        rows = slice(None) if rows is None else rows
        out = top_reference_dsv4._sparse_attn(q[:, rows], kv, layer.attn.attn_sink, topk[:, rows], 0.3)
        return (out * weight[:, rows]).sum()

    plain = torch.autograd.grad(loss(), layer.attn.attn_sink)[0]
    with top_reference_dsv4.token_terms(layer) as sums:
        loss().backward()
    assert torch.allclose(layer.attn.attn_sink.grad, plain, rtol=1e-6, atol=1e-7)
    per_token = [float(torch.autograd.grad(loss(slice(t, t + 1)), layer.attn.attn_sink)[0].norm()) for t in range(5)]
    assert sums == {"attn.attn_sink": pytest.approx(sum(per_token), rel=1e-5)}


def test_dsv4_top_names_of_its_near_cancelling_tensors():
    assert top_reference_dsv4.top_name("model.layers.2.attn.attn_sink") == (
        "decoder.layers.2.self_attention.core_attention.attn_sink"
    )
    assert top_reference_dsv4.top_name("model.layers.3.hc_ffn_base") == "decoder.layers.3.mlp_hyper_connection.bias"
    assert top_reference_dsv4.top_name("model.layers.3.hc_ffn_scale") is None
