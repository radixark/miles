from types import SimpleNamespace

import pytest

from miles.backends.megatron_utils.megatron_to_hf.inkling import _qkv_tp_size
from miles.backends.training_utils import parallel


class TestInklingTensorParallelConfig:
    @pytest.mark.parametrize("configured,live,expected", [(3, None, 3), (None, None, 1), (3, 2, 2)])
    def test_qkv_layout_uses_live_parallelism_or_the_explicit_offline_conversion_config(
        self, monkeypatch: pytest.MonkeyPatch, configured: int | None, live: int | None, expected: int
    ) -> None:
        """Offline Inkling export honors checkpoint TP while a live process group remains authoritative."""
        state = None if live is None else SimpleNamespace(tp=SimpleNamespace(size=live))
        monkeypatch.setattr(parallel, "_parallel_state", state)
        args = SimpleNamespace(backend=SimpleNamespace(tensor_model_parallel_size=configured))

        assert _qkv_tp_size(args) == expected
