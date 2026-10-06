import dataclasses
from types import ModuleType, SimpleNamespace

import msgspec
import pytest


@dataclasses.dataclass
class _Parallelism:
    tp_rank: int
    global_rank: int

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _config(model_replica_module: ModuleType, *, tp_rank: int, global_rank: int, quantization: str | None = None):
    return model_replica_module.RolloutEngineRankConfig(
        parallelism=_Parallelism(tp_rank=tp_rank, global_rank=global_rank),
        server_args=SimpleNamespace(quantization=quantization),
    )


class TestShardLayoutKey:
    def test_ranks_that_differ_only_in_their_place_in_the_launch_share_a_layout(
        self, model_replica_module: ModuleType
    ) -> None:
        """Engines launched on different GPUs hold the same rank the same way, so one replica serves them all."""
        first_engine = _config(model_replica_module, tp_rank=0, global_rank=0)
        second_engine = _config(model_replica_module, tp_rank=0, global_rank=8)

        assert first_engine.shard_layout_key == second_engine.shard_layout_key

    def test_ranks_with_another_shard_or_quantization_do_not(self, model_replica_module: ModuleType) -> None:
        """A replica built for one shard or quantization would write wrong bytes into another."""
        config = _config(model_replica_module, tp_rank=0, global_rank=0)

        assert config.shard_layout_key != _config(model_replica_module, tp_rank=1, global_rank=1).shard_layout_key
        assert (
            config.shard_layout_key
            != _config(model_replica_module, tp_rank=0, global_rank=0, quantization="fp8").shard_layout_key
        )


@pytest.mark.parametrize(
    "record_factory", [dataclasses.make_dataclass, msgspec.defstruct], ids=["dataclass", "msgspec"]
)
def test_server_args_drop_fields_this_sglang_does_not_know(
    model_replica_module: ModuleType, monkeypatch: pytest.MonkeyPatch, record_factory
) -> None:
    """An engine on another sglang reports fields this ServerArgs lacks; they must not break the query."""
    server_args_type = record_factory("ServerArgs", [("model_path", str)])
    monkeypatch.setattr(model_replica_module, "ServerArgs", server_args_type)

    server_args = model_replica_module.create_server_args_from_dict({"model_path": "/model", "unknown_field": True})

    assert isinstance(server_args, server_args_type)
    assert server_args.model_path == "/model"
