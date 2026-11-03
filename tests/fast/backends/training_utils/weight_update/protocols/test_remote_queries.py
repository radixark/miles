import dataclasses
from typing import Any

import msgspec
import pytest
from miles.backends.training_utils.weight_update.protocols.utils.rollout_engine_rank_assignment import (
    RolloutEngineRankAssignment,
)


@pytest.mark.parametrize(
    "record_factory", [dataclasses.make_dataclass, msgspec.defstruct], ids=["dataclass", "msgspec"]
)
def test_server_args_from_remote_info_filters_unknown_fields(
    model_replica_module: Any, monkeypatch: pytest.MonkeyPatch, record_factory: Any
) -> None:
    """Remote metadata must construct both supported record types while ignoring unrelated server fields."""
    server_args_type = record_factory("ServerArgs", [("model_path", str)])
    monkeypatch.setattr(model_replica_module, "ServerArgs", server_args_type)

    result = model_replica_module.create_server_args_from_dict({"model_path": "/model", "unknown_field": True})

    assert isinstance(result, server_args_type)
    assert result.model_path == "/model"


class TestQueriesGiveUpOnBrokenCells:
    @pytest.mark.parametrize(
        "query_name",
        [
            "get_server_info",
            "get_parallelism_info target",
            "get_remote_instance_transfer_engine_info target",
            "get_parallelism_info draft",
            "get_remote_instance_transfer_engine_info draft",
        ],
    )
    @pytest.mark.parametrize("failure", ["error", "timeout"])
    def test_each_query_failure_excludes_the_cell_from_both_runner_roles(
        self, p2p_sender: Any, make_rollout_api: Any, make_bucket: Any, query_name: str, failure: str
    ) -> None:
        """Every metadata request is bounded and failed cells lose target and draft shards without reassignment."""
        protocol = p2p_sender.make_protocol()
        protocol.args.update_weight_engine_request_timeout = 0.02
        speculative_args = {"speculative_algorithm": "EAGLE", "speculative_draft_model_path": "/model"}
        broken = make_rollout_api("broken", gpu_count=1, speculative_args=speculative_args)
        healthy = make_rollout_api("healthy", gpu_count=1, speculative_args=speculative_args)
        if failure == "error":
            broken.request_failures[query_name] = ConnectionError("engine gone")
        else:
            broken.hanging_requests.add(query_name)

        p2p_sender.connect(protocol, [broken, healthy])
        p2p_sender.begin_sync(protocol, weight_version=1)
        protocol.send_bucket(make_bucket("hf.w", "hf.mtp"))
        protocol.after_base_weights()

        assert protocol.cell_updaters_of_cell_id["broken"].is_errored
        assert not protocol.cell_updaters_of_cell_id["healthy"].is_errored
        assert p2p_sender.transfer_engine.written_sessions() == [healthy.session_id(0), healthy.session_id(0, "draft")]

    def test_an_already_errored_cell_is_not_queried(
        self,
        p2p_sender: Any,
        make_rollout_api: Any,
        make_cell_updater: Any,
        model_replica_module: Any,
        p2p_protocol: Any,
        mooncake_module: Any,
    ) -> None:
        """A cell removed earlier contributes neither metadata nor new requests at any query stage."""
        api = make_rollout_api("lost", gpu_count=1)
        cell = make_cell_updater("lost", api)
        cell.mark_errored(RuntimeError("lost"))
        assignments = [RolloutEngineRankAssignment(rollout_engine_rank=0, rollout_engine_indices=[0])]

        assert model_replica_module.query_runner_roles({cell.cell_id: cell}, [cell.cell_id], assignments, "all") == ()
        assert (
            model_replica_module.query_rollout_engine_rank_configs(
                {cell.cell_id: cell}, [cell.cell_id], assignments, "target"
            )
            == {}
        )
        assert p2p_protocol.P2PTransport(mooncake_module.MooncakeTransport()).connect(
            {cell.cell_id: cell}, [cell.cell_id], assignments, ["target"]
        ) == {"target": {0: []}}
        assert api.calls == []
