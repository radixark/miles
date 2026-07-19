from unittest.mock import Mock

from miles.backends.sglang_utils.sglang_engine import SGLangEngine


def test_update_weights_from_disk_releases_sglang_allocator_cache():
    engine = object.__new__(SGLangEngine)
    engine._make_request = Mock(return_value="ok")

    result = engine.update_weights_from_disk("/nvme/model", load_format="auto", weight_version="3")

    assert result == "ok"
    engine._make_request.assert_called_once_with(
        "update_weights_from_disk",
        {
            "model_path": "/nvme/model",
            "torch_empty_cache": True,
            "load_format": "auto",
            "weight_version": "3",
        },
    )
