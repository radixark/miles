import logging
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from miles.backends.megatron_utils import initialize
from miles.backends.training_utils.parallel import GroupInfo, ParallelState


def _parallel_state(
    *,
    tp_rank: int = 0,
    pp_rank: int = 0,
    pp_size: int = 1,
    intra_dp_rank: int = 0,
    intra_dp_size: int = 1,
    intra_dp_cp_rank: int = 0,
    intra_dp_cp_size: int = 1,
    indep_dp_rank: int = 0,
    indep_dp_size: int = 1,
) -> ParallelState:
    intra_dp = GroupInfo(rank=intra_dp_rank, size=intra_dp_size, group=None)
    indep_dp = GroupInfo(rank=indep_dp_rank, size=indep_dp_size, group=None)
    return ParallelState(
        intra_dp=intra_dp,
        intra_dp_cp=GroupInfo(rank=intra_dp_cp_rank, size=intra_dp_cp_size, group=None),
        cp=GroupInfo(rank=0, size=1, group=None),
        tp=GroupInfo(rank=tp_rank, size=max(tp_rank + 1, 1), group=None),
        pp=GroupInfo(rank=pp_rank, size=pp_size, group=None),
        ep=GroupInfo(rank=0, size=1, group=None),
        etp=GroupInfo(rank=0, size=1, group=None),
        indep_dp=indep_dp,
    )


def _patch_state(monkeypatch, state: ParallelState) -> None:
    monkeypatch.setattr(initialize, "get_parallel_state", lambda: state)


class TestLocalReplicaMainRank:
    def test_each_ft_cell_has_one_local_representative(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Independent-DP alive rank does not suppress another cell's local representative."""
        _patch_state(monkeypatch, _parallel_state(indep_dp_rank=2, indep_dp_size=3))

        assert initialize.is_local_replica_megatron_main_rank()

    def test_nonzero_intra_dp_rank_is_not_a_local_representative(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ordinary data-parallel copies emit only one snapshot per cell."""
        _patch_state(monkeypatch, _parallel_state(intra_dp_rank=1, intra_dp_size=2))

        assert not initialize.is_local_replica_megatron_main_rank()


class TestSetRandomSeedFromArgs:
    def test_it_reseeds_rngs_and_only_rank_zero_logs(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Repeated setup reproduces CPU RNG streams while only rank zero announces the seed."""
        _patch_state(monkeypatch, _parallel_state())
        monkeypatch.setattr(initialize.tensor_parallel, "model_parallel_cuda_manual_seed", lambda *args: None)
        args = SimpleNamespace(
            backend=SimpleNamespace(
                rank=0,
                seed=1729,
                data_parallel_random_init=True,
                te_rng_tracker=False,
                inference_rng_tracker=True,
            )
        )

        with caplog.at_level(logging.INFO, logger=initialize.__name__):
            initialize.set_random_seed_from_args(args)
            first_values = (random.random(), np.random.random(), torch.rand(1).item())
            args.backend.rank = 1
            initialize.set_random_seed_from_args(args)
            second_values = (random.random(), np.random.random(), torch.rand(1).item())

        assert second_values == first_values
        assert [record.message for record in caplog.records] == ["> setting random seeds to 1729 ..."]


def test_intra_mode_first_inner_rank_is_main_rank(monkeypatch):
    """INTRA mode: last pp stage, tp rank 0, intra_dp_cp rank 0 is the main rank."""
    _patch_state(monkeypatch, _parallel_state(pp_rank=1, pp_size=2, intra_dp_cp_rank=0, intra_dp_cp_size=2))
    assert initialize.is_first_replica_megatron_main_rank() is True


def test_intra_mode_non_first_inner_rank_is_not_main_rank(monkeypatch):
    """INTRA mode: a non-zero intra_dp_cp rank is not the main rank."""
    _patch_state(monkeypatch, _parallel_state(pp_rank=1, pp_size=2, intra_dp_cp_rank=1, intra_dp_cp_size=2))
    assert initialize.is_first_replica_megatron_main_rank() is False


def test_indep_mode_first_replica_first_inner_is_main_rank(monkeypatch):
    """INDEP mode: first replica plus first inner rank yields the main rank."""
    _patch_state(
        monkeypatch,
        _parallel_state(
            pp_rank=1, pp_size=2, intra_dp_cp_rank=0, intra_dp_cp_size=2, indep_dp_rank=0, indep_dp_size=2
        ),
    )
    assert initialize.is_first_replica_megatron_main_rank() is True


def test_indep_mode_second_replica_main_rank_is_not_first_replica(monkeypatch):
    """INDEP regression: second replica's main rank is not the first-replica main rank."""
    _patch_state(
        monkeypatch,
        _parallel_state(
            pp_rank=1, pp_size=2, intra_dp_cp_rank=0, intra_dp_cp_size=2, indep_dp_rank=1, indep_dp_size=2
        ),
    )
    assert initialize.is_first_replica_megatron_main_rank() is False


def test_non_zero_tp_rank_is_not_main_rank(monkeypatch):
    """A non-zero tp rank is never the main rank even on the first replica's first inner rank."""
    _patch_state(monkeypatch, _parallel_state(tp_rank=1, pp_rank=1, pp_size=2))
    assert initialize.is_first_replica_megatron_main_rank() is False


def test_non_last_pp_stage_is_not_main_rank(monkeypatch):
    """A pp rank that is not the last pipeline stage is never the main rank."""
    _patch_state(monkeypatch, _parallel_state(tp_rank=0, pp_rank=0, pp_size=2))
    assert initialize.is_first_replica_megatron_main_rank() is False


class TestCustomInitializationRequest:
    @pytest.mark.parametrize("fails", [False, True])
    @pytest.mark.parametrize("request_kind", ["fresh", "recovery", "absent"])
    def test_init_passes_only_the_current_request_without_mutating_backend(
        self, monkeypatch: pytest.MonkeyPatch, fails: bool, request_kind: str
    ) -> None:
        """Real init hooks receive transient load inputs even when the callback raises."""
        from tests.fast.fixtures.args_fixtures import parse_megatron_test_config

        from miles.backends.megatron_utils.checkpoint_request import CHECKPOINT_LOAD_FIELDS, MegatronCkptLoadConfig
        from miles.utils.args.custom_function import CustomFunctionConfig
        from miles.utils.args.custom_view import ImmutableNamespace
        from miles.utils.args.trainer_utils import compute_trainer_config
        from miles.utils.function_registry import function_registry

        config = parse_megatron_test_config()
        args = compute_trainer_config(config, config.raw_megatron.trainers[0]).model_copy(
            update={"custom_megatron_init_path": CustomFunctionConfig(path="test.transient_init")}
        )
        request = MegatronCkptLoadConfig.for_peer_recovery()
        if request_kind == "fresh":
            request = request.model_copy(
                update={
                    "load": "original-hf",
                    "resume_from_ckpt": False,
                    "finetune": True,
                    "no_load_optim": True,
                    "no_load_rng": True,
                }
            )
        elif request_kind == "absent":
            request = None
        with args.backend.mutable():
            args.backend.rank = 0
        assert args.backend.data_parallel_size == 1
        original = vars(args.backend).copy()
        seen = []

        for name in (
            "set_args",
            "_initialize_distributed",
            "create_indep_dp_group",
            "create_megatron_parallel_state",
            "set_parallel_state",
            "set_random_seed_from_args",
            "register_hf_config_aliases",
            "_build_tokenizer",
            "init_num_microbatches_calculator",
        ):
            monkeypatch.setattr(initialize, name, lambda *args, **kwargs: None)
        monkeypatch.setattr(initialize.dist, "get_rank", lambda: 0)
        monkeypatch.setattr(initialize.dist, "get_world_size", lambda: 1)

        def check_request(view: ImmutableNamespace) -> None:
            seen.append(view)
            assert CHECKPOINT_LOAD_FIELDS.isdisjoint(vars(args.backend))
            if request is None:
                assert CHECKPOINT_LOAD_FIELDS.isdisjoint(dict(view))
            else:
                for key, value in request.model_dump().items():
                    assert dict(view)[key] == value
            if fails:
                raise RuntimeError("initialization hook failed")

        with function_registry.temporary("test.transient_init", check_request):
            if fails:
                with pytest.raises(RuntimeError, match="initialization hook failed"):
                    initialize.init(args, ckpt_load_config=request)
            else:
                initialize.init(args, ckpt_load_config=request)
        assert len(seen) == 1
        assert vars(args.backend) == original
