from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from miles.backends.megatron_utils.update_weight.update_weight_from_distributed import broadcast


@pytest.mark.parametrize(
    "ip,expected_prefix",
    [
        ("::1", "tcp://[::1]:"),
        ("127.0.0.1", "tcp://127.0.0.1:"),
    ],
)
def test_connect_rollout_engines_builds_valid_init_method(ip, expected_prefix):
    engine = MagicMock()
    with (
        patch.object(broadcast, "ray") as mock_ray,
        patch.object(broadcast, "init_process_group") as mock_init_pg,
    ):
        mock_ray._private.services.get_node_ip_address.return_value = ip
        broadcast.connect_rollout_engines_from_distributed(
            Namespace(rollout_num_gpus_per_engine=1),
            group_name="test",
            rollout_engines=[engine],
        )

    assert mock_init_pg.call_args.kwargs["init_method"].startswith(expected_prefix)
