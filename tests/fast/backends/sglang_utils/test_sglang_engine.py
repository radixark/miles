import inspect

import pytest


class _RecordingApiClient:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def __getattr__(self, name: str):
        def method(**kwargs):
            self.calls.append((name, kwargs))
            return {"called": name}

        return method


def _make_engine():
    pytest.importorskip("sglang")
    from miles.backends.sglang_utils.sglang_engine import SGLangEngine

    engine = SGLangEngine.__new__(SGLangEngine)
    engine.api_client = _RecordingApiClient()
    return engine


def test_engine_methods_forward_to_the_api_client():
    """SGLangEngine is a thin shell over SGLangApiClient."""
    engine = _make_engine()

    assert engine.update_weights_from_tensor(serialized_named_tensors=["a"]) == {
        "called": "update_weights_from_tensor"
    }
    assert engine.health_generate() == {"called": "health_generate"}
    assert engine.check_weights(action="snapshot") == {"called": "check_weights"}

    assert [name for name, _kwargs in engine.api_client.calls] == [
        "update_weights_from_tensor",
        "health_generate",
        "check_weights",
    ]


def _forwarding_method_names() -> list[str]:
    pytest.importorskip("sglang")
    from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
    from miles.backends.sglang_utils.sglang_engine import SGLangEngine

    client_methods = {name for name in vars(SGLangApiClient) if not name.startswith("_")}
    return sorted(name for name in vars(SGLangEngine) if name in client_methods and name != "pull_weights")


@pytest.mark.parametrize("method_name", _forwarding_method_names())
def test_shell_forwards_every_argument_verbatim(method_name):
    """No shell drops, renames or defaults-over any of the arguments it forwards."""
    engine = _make_engine()
    signature = inspect.signature(getattr(engine, method_name))
    kwargs = {
        name: param.default if param.default is not inspect.Parameter.empty else f"<{name}>"
        for name, param in signature.parameters.items()
    }

    getattr(engine, method_name)(**kwargs)

    assert engine.api_client.calls == [(method_name, kwargs)]


def test_pull_weights_supplies_the_checkpoint_dirs_from_args():
    """The client is args-free, so the engine shell resolves both dirs before delegating."""
    engine = _make_engine()
    engine.args = type(
        "Args", (), {"update_weight_local_checkpoint_dir": "/local", "update_weight_disk_dir": "/shared"}
    )()

    engine.pull_weights(target_version=5)

    assert engine.api_client.calls == [
        ("pull_weights", {"target_version": 5, "local_checkpoint_dir": "/local", "source_dir": "/shared"}),
    ]
