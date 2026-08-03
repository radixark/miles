from __future__ import annotations

import shlex
import sys

import pytest
from tests.fast.backends.sglang_utils.conftest import make_engine_args, tiny_model_path

pytest.importorskip("sglang")

from miles.backends.sglang_utils.server_args_utils import parse_server_args_argv
from miles.backends.sglang_utils.sglang_engine import compute_engine_launch_cmd


def _cmd(
    *,
    worker_type: str = "regular",
    args=None,
    addr_overrides: dict | None = None,
    base_gpu_id: int = 0,
    **kwargs,
) -> str:
    addr_and_ports = dict(
        host="10.0.0.1",
        port=30000,
        nccl_port=20031,
        engine_info_bootstrap_port=20033,
        gated_launch_port=20034,
        dist_init_addr="10.0.0.1:20000",
        disaggregation_bootstrap_port=None,
    )
    addr_and_ports.update(addr_overrides or {})
    return compute_engine_launch_cmd(
        args or make_engine_args(),
        node_rank=0,
        worker_type=worker_type,
        base_gpu_id=base_gpu_id,
        # ServerArgs probes the local accelerator when no device is given, which a CPU-only
        # CI runner cannot answer. Production resolves it to the engine's own device the same way.
        sglang_overrides={"device": "cuda"},
        num_gpus_per_engine=1,
        dist_init_addr=addr_and_ports["dist_init_addr"],
        nccl_port=addr_and_ports["nccl_port"],
        host=addr_and_ports["host"],
        port=addr_and_ports["port"],
        disaggregation_bootstrap_port=addr_and_ports["disaggregation_bootstrap_port"],
        engine_info_bootstrap_port=addr_and_ports["engine_info_bootstrap_port"],
        gated_launch_port=addr_and_ports["gated_launch_port"],
        **kwargs,
    )


class TestComputeEngineLaunchCmd:
    def test_the_command_launches_sglang_with_the_allocated_addressing(self):
        """The rendered launch_server command carries the addr map."""
        tokens = shlex.split(_cmd())
        assert tokens[:3] == [sys.executable, "-m", "sglang.launch_server"]
        parsed = parse_server_args_argv(tokens[3:])
        assert parsed.host == "10.0.0.1" and parsed.port == 30000
        assert parsed.dist_init_addr == "10.0.0.1:20000"
        assert parsed.gated_launch_port == 20034
        assert parsed.model_path == str(tiny_model_path())

    def test_a_bracketed_v6_host_is_stripped_for_the_server_but_kept_in_dist_addr(self):
        """sglang binds a bare v6 host while the rendezvous addr stays bracketed."""
        cmd = _cmd(addr_overrides=dict(host="[fd00::2]", port=31007, dist_init_addr="[fd00::1]:15003"))
        parsed = parse_server_args_argv(shlex.split(cmd)[3:])
        assert parsed.host == "fd00::2"
        assert parsed.dist_init_addr == "[fd00::1]:15003"

    def test_a_prefill_command_carries_the_bootstrap_port(self):
        """PD-disaggregation prefill flags survive into the command."""
        cmd = _cmd(worker_type="prefill", addr_overrides=dict(disaggregation_bootstrap_port=20090))
        parsed = parse_server_args_argv(shlex.split(cmd)[3:])
        assert parsed.disaggregation_mode == "prefill"
        assert parsed.disaggregation_bootstrap_port == 20090

    def test_the_command_carries_the_api_key_from_args(self):
        """--sglang-api-key reaches the server through the generic passthrough."""
        cmd = _cmd(args=make_engine_args(sglang_api_key="secret"))
        parsed = parse_server_args_argv(shlex.split(cmd)[3:])
        assert parsed.api_key == "secret"


class TestLoraTargetModules:
    @staticmethod
    def _parsed_lora_targets(target_modules: list[str]):
        args = make_engine_args(lora_rank=16, target_modules=target_modules)
        return parse_server_args_argv(shlex.split(_cmd(args=args))[3:]).lora_target_modules

    def test_spellable_targets_are_named_one_by_one(self):
        """Naming the exact modules keeps SGLang from allocating adapter buffers for the rest."""
        targets = self._parsed_lora_targets(["layers.*.self_attention.linear_qkv"])

        assert sorted(targets) == ["k_proj", "q_proj", "v_proj"]

    def test_gdn_attention_targets_are_named_one_by_one(self):
        """Qwen3.5 GDN adapters must reach the engine as the exact fused slices, not as the
        auto-detecting shorthand that would cover every compatible module instead."""
        targets = self._parsed_lora_targets(["layers.*.self_attention.in_proj"])

        assert sorted(targets) == ["in_proj_ba", "in_proj_qkvz"]

    def test_an_inkling_checkpoint_asks_sglang_to_discover_the_names(self, monkeypatch: pytest.MonkeyPatch):
        """Inkling exposes module names the megatron-to-HF mapping cannot produce, so it is the
        one family that hands SGLang the shorthand instead of naming its targets."""
        monkeypatch.setattr(
            "miles.backends.sglang_utils.sglang_engine.sglang_lora_target_all_sentinel", lambda _args: True
        )

        targets = self._parsed_lora_targets(["layers.*.self_attention.linear_qkv"])

        assert set(targets) == {"all"}

    def test_a_multi_lora_inkling_launch_still_names_its_targets(self, monkeypatch: pytest.MonkeyPatch):
        """Several adapters share one slot budget here, so discovering every compatible module
        sizes that budget off the base model instead of off what the adapters fill."""
        monkeypatch.setattr(
            "miles.backends.sglang_utils.sglang_engine.sglang_lora_target_all_sentinel", lambda _args: True
        )
        args = make_engine_args(
            lora_rank=16,
            target_modules=["layers.*.self_attention.linear_qkv"],
            multi_lora=True,
            multi_lora_n_adapters=4,
        )

        targets = parse_server_args_argv(shlex.split(_cmd(args=args))[3:]).lora_target_modules

        assert sorted(targets) == ["k_proj", "q_proj", "v_proj"]

    def test_asking_for_every_module_is_still_honoured(self):
        """SGLang accepts the shorthand as a target name, so a run that spelled it out itself
        is not the substitution this refuses."""
        targets = self._parsed_lora_targets(["all"])

        assert set(targets) == {"all"}
