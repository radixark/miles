from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.fast.optimizers.nvme_setup_fixtures import make_nvme_setup_args, nvme_setup_chain

from miles_plugins.optimizers.nvme_stream import setup_optimizer_state_streaming

__all__ = ["nvme_setup_chain"]


class TestNvmeSetupBackendOwnership:
    @pytest.mark.parametrize("no_load_optim", [False, True])
    def test_missing_streamed_checkpoint_obeys_the_backend_optimizer_load_policy(
        self, tmp_path: Path, nvme_setup_chain: SimpleNamespace, no_load_optim: bool
    ) -> None:
        """Fresh optimizer state is accepted only when the backend explicitly disables optimizer restoration."""
        args = make_nvme_setup_args(directory=tmp_path / "scratch", no_load_optim=no_load_optim)

        setup_optimizer_state_streaming(args, nvme_setup_chain)

        [optimizer] = nvme_setup_chain.chained_optimizers
        store = optimizer._nvme_state_store
        if no_load_optim:
            assert store.load_from(str(tmp_path / "missing-checkpoint")) is False
        else:
            with pytest.raises(AssertionError, match="resuming would restart Adam from zero"):
                store.load_from(str(tmp_path / "missing-checkpoint"))
