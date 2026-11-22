import pytest
from tests.fast.backends.sglang_utils.test_compute_server_args import compute, make_args


class TestLoraCpuBackup:
    @pytest.mark.parametrize(
        "rank,adapter,train_only,colocate,backup,expected",
        [
            (8, None, False, True, True, True),
            (0, "/pretrained/adapter", False, True, True, True),
            (0, None, False, True, True, False),
            (8, None, False, False, True, False),
            (8, None, False, True, False, False),
            (8, None, True, True, True, False),
        ],
    )
    def test_only_colocated_rollout_adapters_request_the_engine_cpu_weight_backup(
        self,
        rank: int,
        adapter: str | None,
        train_only: bool,
        colocate: bool,
        backup: bool,
        expected: bool,
    ) -> None:
        """The actual SGLang argument consumer enables host weight backup only for the requested rollout mode."""
        args = make_args(
            lora_rank=rank,
            lora_adapter_path=adapter,
            lora_train_only=train_only,
            colocate=colocate,
            lora_base_cpu_backup=backup,
        )

        server_args = compute(args)

        assert bool(server_args.get("enable_weights_cpu_backup")) is expected
        if adapter is not None:
            assert server_args["max_lora_rank"] == 1
