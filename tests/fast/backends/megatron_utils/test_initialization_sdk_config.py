import pytest
import torch
from tests.fast.backends.megatron_utils.initialization_sdk_fixtures import (
    _initialization_args,
    _InitializationSdk,
    initialization_sdk,
)

from miles.backends.megatron_utils.megatron_config import MegatronArgsNamespace

__all__ = ["initialization_sdk"]


class TestBackendInitializationSdk:
    @pytest.mark.parametrize(
        "experimental,overlap,deterministic", [(True, False, False), (False, True, False), (False, False, True)]
    )
    def test_real_initialization_passes_backend_to_tokenizer_and_batch_calculator(
        self, initialization_sdk: _InitializationSdk, experimental: bool, overlap: bool, deterministic: bool
    ) -> None:
        """Real initialization uses backend storage for SDK setup and derives tokenizer state within a temporary mutation scope."""
        args = _initialization_args(
            enable_experimental=experimental,
            tp_comm_overlap=overlap,
            deterministic_mode=deterministic,
            rank=0,
            seed=71,
            data_parallel_random_init=False,
            te_rng_tracker=False,
            inference_rng_tracker=False,
            cp_comm_type="p2p",
            indep_dp=False,
            rampup_batch_size=[4, 8, 16],
            global_batch_size=12,
            micro_batch_size=3,
            data_parallel_size=2,
            decrease_batch_size_if_needed=True,
            debug_deterministic_collective=False,
            custom_megatron_init_path=None,
        )

        def build_tokenizer(backend: MegatronArgsNamespace) -> None:
            assert backend is args.backend
            backend.padded_vocab_size = 321

        initialization_sdk.tokenizer.side_effect = build_tokenizer
        initialization_sdk.module.init(args)

        initialization_sdk.installed_args.assert_called_once_with(args.backend)
        initialization_sdk.tokenizer.assert_called_once_with(args.backend)
        initialization_sdk.microbatches.assert_called_once_with(0, [4, 8, 16], 12, 3, 2, True)
        if experimental:
            initialization_sdk.experimental.assert_called_once_with(True)
        else:
            initialization_sdk.experimental.assert_not_called()
        assert initialization_sdk.communicators.call_count == int(overlap)
        if deterministic:
            initialization_sdk.deterministic.assert_called_once_with(True, warn_only=False)
            assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
            assert not torch.utils.deterministic.fill_uninitialized_memory
        else:
            initialization_sdk.deterministic.assert_not_called()
        assert args.backend.padded_vocab_size == 321
        with pytest.raises(AttributeError, match="immutable"):
            args.backend.padded_vocab_size = 99

    def test_tokenizer_sdk_failure_restores_backend_immutability(self, initialization_sdk: _InitializationSdk) -> None:
        """A failed tokenizer setup cannot leave the backend mutable for the rest of the worker lifecycle."""
        args = _initialization_args(
            enable_experimental=False,
            tp_comm_overlap=False,
            deterministic_mode=False,
            rank=0,
            seed=71,
            data_parallel_random_init=False,
            te_rng_tracker=False,
            inference_rng_tracker=False,
            cp_comm_type="p2p",
            indep_dp=False,
        )
        initialization_sdk.tokenizer.side_effect = RuntimeError("tokenizer SDK failed")

        with pytest.raises(RuntimeError, match="tokenizer SDK failed"):
            initialization_sdk.module.init(args)

        initialization_sdk.microbatches.assert_not_called()
        with pytest.raises(AttributeError, match="immutable"):
            args.backend.padded_vocab_size = 99

    def test_independent_cells_reject_backend_data_parallel_replicas_before_tokenizer(
        self, initialization_sdk: _InitializationSdk
    ) -> None:
        """Independent cells cannot also use nontrivial backend data parallelism."""
        args = _initialization_args(indep_dp=True, data_parallel_size=2)

        with pytest.raises(AssertionError):
            initialization_sdk.module.init(args)

        initialization_sdk.tokenizer.assert_not_called()
        initialization_sdk.microbatches.assert_not_called()

    @pytest.mark.parametrize("overlap", [False, True])
    def test_deterministic_collectives_reject_overlapped_gradient_reduction(
        self, initialization_sdk: _InitializationSdk, overlap: bool
    ) -> None:
        """Deterministic collectives allow synchronous gradients and reject the backend overlap option."""
        args = _initialization_args(debug_deterministic_collective=True, overlap_grad_reduce=overlap)

        if overlap:
            with pytest.raises(AssertionError, match="deterministic collectives require synchronous grad sync"):
                initialization_sdk.module.init(args)
        else:
            initialization_sdk.module.init(args)

        initialization_sdk.tokenizer.assert_called_once_with(args.backend)
        initialization_sdk.microbatches.assert_called_once_with(0, [4, 8, 16], 12, 3, 2, True)
