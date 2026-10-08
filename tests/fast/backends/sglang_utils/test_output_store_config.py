from types import SimpleNamespace

import pytest

from miles.backends.sglang_utils.output_store_config import validate_output_store_args


def _args(**overrides: object) -> SimpleNamespace:
    values = dict(
        sglang_output_store_backend="mooncake",
        sglang_output_store_backend_extra_config=None,
        object_store_backend="mooncake",
        mooncake_store_init_kwargs={"master_server_address": "10.0.0.1:50051"},
        mooncake_replica_num=1,
        prefill_num_servers=None,
    )
    return SimpleNamespace(**(values | overrides))


def test_a_run_with_a_mooncake_object_store_passes():
    validate_output_store_args(_args())


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"object_store_backend": "ray"}, "--object-store-backend mooncake"),
        ({"prefill_num_servers": 1}, "PD disaggregation"),
        ({"sglang_output_store_backend_extra_config": '{"replica_num": 3}'}, "contradicts"),
        ({"sglang_output_store_backend_extra_config": "[1]"}, "JSON object"),
    ],
)
def test_a_run_the_rollout_executor_could_not_read_from_fails_at_startup(overrides, match):
    with pytest.raises(ValueError, match=match):
        validate_output_store_args(_args(**overrides))


def test_nothing_is_checked_while_the_backend_is_off():
    validate_output_store_args(_args(sglang_output_store_backend="none", object_store_backend="ray"))
