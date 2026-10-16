from __future__ import annotations

import argparse
import dataclasses
from types import SimpleNamespace

import pytest

pytest.importorskip("sglang")

from sglang.srt.server_args import ServerArgs

from miles.backends.sglang_utils import arguments
from miles.backends.sglang_utils.arguments import add_sglang_arguments, collect_eval_sglang_overrides, validate_args
from miles.utils.workers.argv_utils import _record_field_names


def _sglang_flags() -> set[str]:
    parser = add_sglang_arguments(argparse.ArgumentParser())
    return {option for action in parser._actions for option in action.option_strings}


def _parse_sglang_args(argv: list[str]) -> argparse.Namespace:
    return add_sglang_arguments(argparse.ArgumentParser()).parse_args(argv)


class TestSglangModelRoutersDefault:
    def test_parsing_without_model_routers_sets_none(self):
        """Parsing without multi-policy routers exposes a safe None default."""
        args = _parse_sglang_args([])

        assert args.sglang_model_routers is None


class TestAllocatorOwnedServerArgs:
    def test_the_launch_gate_port_is_not_exposed_on_the_cli(self):
        """The gate port comes from the addr allocator, so a flag for it could only point the engine elsewhere."""
        flags = _sglang_flags()

        assert "--sglang-gated-launch-port" not in flags
        assert "--eval-sglang-gated-launch-port" not in flags

    def test_a_tunable_server_arg_is_still_exposed(self):
        """The skip list must stay narrow: ordinary ServerArgs fields remain reachable from the cli."""
        assert "--sglang-mem-fraction-static" in _sglang_flags()

    def test_a_launch_gate_port_flag_is_a_hard_cli_error(self):
        """Passing a gate port on the command line must fail loudly instead of being accepted and ignored."""
        for flag in ("--sglang-gated-launch-port", "--eval-sglang-gated-launch-port"):
            with pytest.raises(SystemExit):
                _parse_sglang_args([flag, "13007"])

    def test_parsing_leaves_no_launch_gate_port_attribute_on_the_namespace(self):
        """The engine copies every args.sglang_<field> onto ServerArgs, so a parsed attribute would travel to it."""
        args = _parse_sglang_args([])

        assert not hasattr(args, "sglang_gated_launch_port")
        assert not hasattr(args, "eval_sglang_gated_launch_port")
        assert "gated_launch_port" not in collect_eval_sglang_overrides(args)

    def test_the_other_allocator_owned_server_args_stay_off_the_cli(self):
        """The gate port joins the existing skip list instead of replacing the endpoint entries already there."""
        flags = _sglang_flags()

        for flag in ("--sglang-port", "--sglang-nccl-port", "--sglang-base-gpu-id", "--eval-sglang-port"):
            assert flag not in flags

    def test_the_skipped_launch_gate_port_names_a_real_server_args_field(self):
        """A renamed upstream field would leave the skip entry stale and quietly re-expose the flag."""
        assert "gated_launch_port" in set(_record_field_names(ServerArgs))


@pytest.mark.parametrize("loss_type", ["score_centering", "policy_loss", "sft_loss", None])
@pytest.mark.parametrize("explicit_fp32", [False, True])
def test_score_centering_enables_fp32_lm_head(loss_type: str | None, explicit_fp32: bool) -> None:
    args = _parse_sglang_args(["--sglang-enable-fp32-lm-head"] if explicit_fp32 else [])
    args.loss_type = loss_type
    args.rollout_num_gpus_per_engine = 1
    args.true_on_policy_mode = False
    args.use_session_server = False

    validate_args(args)

    assert args.sglang_enable_fp32_lm_head is (explicit_fp32 or loss_type == "score_centering")


class TestOutputStoreArgs:
    def test_miles_declares_the_flags_and_derives_the_engine_args(self):
        """Miles sets both engine args from its own flags, so neither passes through from the cli."""
        flags = _sglang_flags()

        assert {"--sglang-output-store-backend", "--sglang-output-store-client-config"} <= flags
        for flag in (
            "--sglang-output-store-backend-extra-config",
            "--eval-sglang-output-store-backend",
            "--eval-sglang-output-store-backend-extra-config",
        ):
            assert flag not in flags
        args = _parse_sglang_args(["--sglang-output-store-client-config", '{"local_buffer_size": "4gb"}'])
        assert args.sglang_output_store_backend == "none"
        assert args.sglang_output_store_client_config == {"local_buffer_size": "4gb"}

    @staticmethod
    def _args(**overrides: object) -> SimpleNamespace:
        values = dict(
            sglang_output_store_backend="mooncake",
            sglang_output_store_client_config={"local_buffer_size": "4gb", "chunk_bytes": 1 << 20},
            object_store_backend="mooncake",
            use_rollout_routing_replay=True,
            use_rollout_indexer_replay=False,
            use_sampling_support_replay=False,
            prefill_num_servers=None,
            rollout_external=False,
        )
        return SimpleNamespace(**(values | overrides))

    @pytest.fixture
    def sglang_with_output_store(self, monkeypatch):
        server_args_type = dataclasses.make_dataclass("ServerArgs", [("output_store_backend", str)])
        monkeypatch.setattr(arguments, "ServerArgs", server_args_type)
        monkeypatch.delenv("MILES_USE_LEGACY_ROLLOUT_V1", raising=False)

    def test_a_run_whose_readers_can_reach_the_store_passes(self, sglang_with_output_store):
        arguments._validate_output_store_args(self._args())

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"object_store_backend": "ray"}, "--object-store-backend mooncake"),
            ({"use_rollout_routing_replay": False}, "enable routing, indexer or sampling-support replay"),
            ({"prefill_num_servers": 1}, "PD disaggregation"),
            ({"rollout_external": True}, "engines Miles launches"),
            ({"sglang_output_store_client_config": None}, "local_buffer_size"),
            ({"sglang_output_store_client_config": {"chunk_bytes": 1 << 20}}, "local_buffer_size"),
            # Each derived key has one owner: a second value would split the engines from the readers.
            (
                {"sglang_output_store_client_config": {"local_buffer_size": "4gb", "protocol": "rdma"}},
                r"got \['protocol'\]",
            ),
            (
                {"sglang_output_store_client_config": {"local_buffer_size": "4gb", "local_buffer": 1}},
                r"got \['local_buffer'\]",
            ),
        ],
    )
    def test_an_unsupported_run_fails_at_startup(self, sglang_with_output_store, overrides, match):
        with pytest.raises(AssertionError, match=match):
            arguments._validate_output_store_args(self._args(**overrides))

    def test_the_legacy_rollout_fails_at_startup(self, sglang_with_output_store, monkeypatch):
        monkeypatch.setenv("MILES_USE_LEGACY_ROLLOUT_V1", "1")

        with pytest.raises(AssertionError, match="MILES_USE_LEGACY_ROLLOUT_V1"):
            arguments._validate_output_store_args(self._args())

    def test_an_sglang_without_the_output_store_fails_at_startup(self, monkeypatch):
        monkeypatch.setattr(arguments, "ServerArgs", dataclasses.make_dataclass("ServerArgs", [("port", int)]))

        with pytest.raises(AssertionError, match="needs an SGLang with --output-store-backend"):
            arguments._validate_output_store_args(self._args())

    def test_nothing_is_checked_while_the_backend_is_off(self):
        arguments._validate_output_store_args(
            self._args(sglang_output_store_backend="none", object_store_backend="ray")
        )
