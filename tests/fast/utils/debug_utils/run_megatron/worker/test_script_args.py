import dataclasses
from argparse import ArgumentParser
from pathlib import Path

import pytest

from miles.utils.debug_utils.run_megatron.worker.script_args import (
    WORKER_SCRIPT_ARGS_BRIDGE,
    WorkerScriptArgs,
    register_worker_arguments,
)


class TestWorkerScriptArgs:
    @pytest.mark.parametrize("argv,implementation", [([], "miles"), (["--dsa-impl", "megatron"], "megatron")])
    def test_worker_parser_registers_dsa_arguments(self, argv: list[str], implementation: str) -> None:
        parser = ArgumentParser()
        parser.add_argument("--cp-comm-type", nargs="+", default=["p2p"])
        register_worker_arguments(parser)
        args = parser.parse_args(
            [
                "--script-hf-checkpoint",
                "/hf",
                "--script-token-ids-file",
                "/tokens.json",
                "--miles-dsa-topk-backend",
                "flashinfer",
            ]
            + argv
        )
        assert args.dsa_impl == implementation
        assert args.miles_dsa_topk_backend == "flashinfer"
        assert args.cp_comm_type is None
        assert WORKER_SCRIPT_ARGS_BRIDGE.from_namespace(args).hf_checkpoint == Path("/hf")

    def test_frozen(self) -> None:
        args = WorkerScriptArgs(hf_checkpoint=Path("/hf"), token_ids_file=Path("/tokens.json"))
        with pytest.raises(dataclasses.FrozenInstanceError):
            args.role = "critic"  # type: ignore[misc]

    def test_defaults(self) -> None:
        args = WorkerScriptArgs(hf_checkpoint=Path("/hf"), token_ids_file=Path("/tokens.json"))
        assert args.role == "actor"
        assert args.top_k == 0
        assert args.run_backward is False
        assert args.ref_load is None
        assert args.source_patcher_config is None
        assert args.routing_replay_dump_path is None
        assert args.routing_replay_load_path is None

    def test_to_cli_args_roundtrip(self) -> None:
        original = WorkerScriptArgs(
            hf_checkpoint=Path("/hf"),
            token_ids_file=Path("/tokens.json"),
            role="critic",
            run_backward=True,
            top_k=5,
        )
        cli_str = WORKER_SCRIPT_ARGS_BRIDGE.to_cli_args(original)

        import argparse

        parser = argparse.ArgumentParser()
        WORKER_SCRIPT_ARGS_BRIDGE.register_on_parser(parser)
        namespace = parser.parse_args(cli_str.split())
        reconstructed = WORKER_SCRIPT_ARGS_BRIDGE.from_namespace(namespace)

        assert reconstructed == original

    def test_none_fields_omitted(self) -> None:
        args = WorkerScriptArgs(hf_checkpoint=Path("/hf"), token_ids_file=Path("/tokens.json"))
        cli_str = WORKER_SCRIPT_ARGS_BRIDGE.to_cli_args(args)
        assert "--script-ref-load" not in cli_str
        assert "--script-source-patcher-config" not in cli_str
        assert "--script-routing-replay-dump-path" not in cli_str
        assert "--script-routing-replay-load-path" not in cli_str

    def test_bool_serialization(self) -> None:
        args = WorkerScriptArgs(
            hf_checkpoint=Path("/hf"),
            token_ids_file=Path("/tokens.json"),
            run_backward=True,
        )
        cli_str = WORKER_SCRIPT_ARGS_BRIDGE.to_cli_args(args)
        assert "--script-run-backward" in cli_str
