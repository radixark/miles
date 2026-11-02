from argparse import Namespace

import pytest

from miles.backends.fsdp_utils.actor import FSDPTrainRayActor
from miles.backends.fsdp_utils.config import FsdpArgsNamespace
from miles.backends.megatron_utils import initialize
from miles.backends.megatron_utils.megatron_config import MegatronArgsNamespace
from miles.backends.training_utils.torch_native.actor import TorchNativeTrainRayActor


class TestTypedBackendInitialization:
    def test_fsdp_rejects_megatron_before_initializing_actor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """FSDP rejects another backend before initializing actor resources."""

        def unexpected_common(*args: object, **kwargs: object) -> None:
            pytest.fail("Actor initialization started with the wrong backend")

        monkeypatch.setattr(TorchNativeTrainRayActor, "_init_common", unexpected_common)
        actor = object.__new__(FSDPTrainRayActor)

        with pytest.raises(AssertionError):
            FSDPTrainRayActor.init.__wrapped__(
                actor,
                Namespace(backend=MegatronArgsNamespace()),
                "actor",
                indep_dp_info=None,
                indep_dp_store_addr=None,
            )

    def test_megatron_rejects_fsdp_before_installing_global_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Megatron rejects another backend before installing its process global arguments."""

        def unexpected_set_args(*args: object, **kwargs: object) -> None:
            pytest.fail("Megatron global arguments were installed with the wrong backend")

        monkeypatch.setattr(initialize, "set_args", unexpected_set_args)

        with pytest.raises(AssertionError):
            initialize.init(args=Namespace(backend=FsdpArgsNamespace()))
