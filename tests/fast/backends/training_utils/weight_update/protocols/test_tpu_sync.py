"""tpu-sync's engine calls must go through the rollout cell updater and the sglang client as they exist."""

from argparse import Namespace
from dataclasses import dataclass, field

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.weight_update.protocols import tpu_sync
from miles.backends.training_utils.weight_update.rollout_cell_updater import create_rollout_cell_updaters


@dataclass(frozen=True)
class _RecordingClient(SGLangApiClient):
    requests: list = field(default_factory=list)

    async def _make_request(self, endpoint, payload=None, *, timeout=None):
        self.requests.append((endpoint, payload))
        return {"success": True}


def test_engine_calls_reach_the_receiver_endpoints():
    """An updater or client method that drifted would otherwise fail the first sync with AttributeError."""
    client = _RecordingClient(server_url="http://engine")
    updaters = list(
        create_rollout_cell_updaters(
            Namespace(ft_components=[], update_weight_engine_request_timeout=10.0),
            [client],
            ["cell-0"],
        ).values()
    )

    tpu_sync._call_engines(
        updaters,
        "tpu_sync.init_receiver",
        lambda updater: dict(
            controller_address="10.0.0.1:1234",
            job_name=tpu_sync._engine_job_name(updater.cell_id),
            weight_shapes={"model.norm.weight": [8]},
            parallelism=8,
        ),
    )
    tpu_sync._call_engines(updaters, "tpu_sync.install", lambda updater: {})

    assert client.requests == [
        (
            "plugin_weight_update",
            {
                "handler": "tpu_sync.init_receiver",
                "payload": {
                    "controller_address": "10.0.0.1:1234",
                    "job_name": "engine-cell-0",
                    "weight_shapes": {"model.norm.weight": [8]},
                    "parallelism": 8,
                },
                # miles keeps or flushes KV in its own pause step
                "flush_cache": False,
            },
        ),
        ("plugin_weight_update", {"handler": "tpu_sync.install", "payload": {}, "flush_cache": False}),
    ]
