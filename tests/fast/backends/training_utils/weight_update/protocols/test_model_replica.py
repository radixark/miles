import dataclasses
from types import ModuleType, SimpleNamespace

import msgspec
import pytest
import torch


@dataclasses.dataclass
class _Parallelism:
    tp_rank: int
    global_rank: int

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class _ServerArgs:
    quantization: str | None = None
    moe_runner_backend: str = "auto"
    enable_dp_lm_head: bool = False
    port: int = 30000
    mem_fraction_static: float = 0.8

    def __getattr__(self, name: str) -> None:
        # the other server args a replica key reads, all unset
        if name.startswith("_"):
            raise AttributeError(name)
        return None


def _config(model_replica_module: ModuleType, *, tp_rank: int, global_rank: int, **server_args):
    return model_replica_module.RolloutEngineRankConfig(
        parallelism=_Parallelism(tp_rank=tp_rank, global_rank=global_rank), server_args=_ServerArgs(**server_args)
    )


class TestShardLayoutKey:
    def test_ranks_that_differ_only_in_their_launch_or_serving_share_a_layout(
        self, model_replica_module: ModuleType
    ) -> None:
        """Engines on other GPUs, ports or memory settings, such as the two PD roles, hold the same rank the same way,
        so one replica serves them all."""
        first_engine = _config(model_replica_module, tp_rank=0, global_rank=0, port=30000, mem_fraction_static=0.8)
        second_engine = _config(model_replica_module, tp_rank=0, global_rank=8, port=30001, mem_fraction_static=0.6)

        assert first_engine.shard_layout_key == second_engine.shard_layout_key

    def test_ranks_with_another_shard_or_sglang_arguments_do_not(self, model_replica_module: ModuleType) -> None:
        """A replica built for one shard, quantization, MoE backend or lm_head sharding would write wrong bytes into
        another."""
        config = _config(model_replica_module, tp_rank=0, global_rank=0)

        assert config.shard_layout_key != _config(model_replica_module, tp_rank=1, global_rank=1).shard_layout_key
        for server_args in ({"quantization": "fp8"}, {"moe_runner_backend": "triton"}, {"enable_dp_lm_head": True}):
            assert (
                config.shard_layout_key
                != _config(model_replica_module, tp_rank=0, global_rank=0, **server_args).shard_layout_key
            )


class TestModelReplicas:
    @pytest.fixture
    def model_replicas_of_width(self, model_replica_module: ModuleType, monkeypatch: pytest.MonkeyPatch):
        """`ModelReplicas` whose replica for tp rank r is a linear layer of `widths[r]` inputs."""

        def make(widths: dict[int, int]):
            monkeypatch.setattr(
                model_replica_module,
                "_build_cpu_replica",
                lambda config, model_path: torch.nn.Linear(widths[config.parallelism.tp_rank], 1, bias=False),
            )
            monkeypatch.setattr(
                model_replica_module, "ParameterMapper", SimpleNamespace(from_model=lambda model: None)
            )
            # CPU CI has no pinned memory
            monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor: tensor)
            return model_replica_module.ModelReplicas(model_path="/model")

        return make

    def test_a_replica_for_another_layout_loads_into_the_shared_buffer(
        self, model_replica_module: ModuleType, model_replicas_of_width
    ) -> None:
        """Writes read the shared buffer, so a later replica's loads must land in it."""
        model_replicas = model_replicas_of_width({0: 4, 1: 4})

        first = model_replicas.get_or_build(_config(model_replica_module, tp_rank=0, global_rank=0))
        second = model_replicas.get_or_build(_config(model_replica_module, tp_rank=1, global_rank=1))

        assert second is not first
        assert (
            second.weight.data_ptr()
            == first.weight.data_ptr()
            == model_replicas.shared_params_dict["weight"].data_ptr()
        )

    def test_a_replica_whose_params_do_not_fit_the_shared_buffer_is_rejected(
        self, model_replica_module: ModuleType, model_replicas_of_width
    ) -> None:
        """A param of another shape cannot alias the shared buffer without loading wrong bytes."""
        model_replicas = model_replicas_of_width({0: 4, 1: 3})
        model_replicas.get_or_build(_config(model_replica_module, tp_rank=0, global_rank=0))

        with pytest.raises(AssertionError, match="in the shared buffer"):
            model_replicas.get_or_build(_config(model_replica_module, tp_rank=1, global_rank=1))


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
