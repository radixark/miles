import sys
from argparse import Namespace
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from miles.backends.training_utils.weight_update.protocol import get_weight_transfer_protocol
from miles.utils.function_registry import function_registry


def _args(**overrides) -> Namespace:
    values = {
        "colocate": False,
        "update_weight_transfer_mode": "broadcast",
        "custom_weight_transfer_protocol_path": None,
    }
    values.update(overrides)
    return Namespace(**values)


def _load_external(path: str):
    return get_weight_transfer_protocol(
        _args(update_weight_transfer_mode="external", custom_weight_transfer_protocol_path=path)
    )


def test_external_factory_rejects_a_target_that_returns_no_protocol():
    with (
        function_registry.temporary("test:target", lambda args: object()),
        pytest.raises(TypeError) as exc_info,
    ):
        _load_external("test:target")

    message = str(exc_info.value)
    assert "--custom-weight-transfer-protocol-path" in message
    assert "'test:target'" in message
    assert "must return a WeightTransferProtocol" in message
    assert "it returned object" in message


@pytest.mark.parametrize(
    ("path", "message_parts"),
    [
        (
            None,
            ["--update-weight-transfer-mode=external requires --custom-weight-transfer-protocol-path"],
        ),
        (
            "missing_attribute",
            ["--custom-weight-transfer-protocol-path", "'missing_attribute'", "must be a dotted import path"],
        ),
    ],
)
def test_external_factory_rejects_an_unusable_path(path, message_parts):
    with pytest.raises(ValueError) as exc_info:
        _load_external(path)

    message = str(exc_info.value)
    for part in message_parts:
        assert part in message


async def _async_target(args):
    return args


@pytest.mark.parametrize(
    ("name", "target", "message_part"),
    [
        ("test:GROUP_NAME", "miles", "did not resolve to a callable"),
        ("test:async_build_protocol", _async_target, "resolved to an async function"),
    ],
)
def test_a_non_synchronous_target_is_rejected_with_the_flag_named(name, target, message_part):
    with (
        function_registry.temporary(name, target),
        pytest.raises(TypeError) as exc_info,
    ):
        _load_external(name)

    message = str(exc_info.value)
    assert "--custom-weight-transfer-protocol-path" in message
    assert f"'{name}'" in message
    assert message_part in message
    assert "synchronous factory" in message


def test_built_in_modes_still_construct_their_protocols(monkeypatch):
    broadcast = ModuleType("miles.backends.training_utils.weight_update.protocols.broadcast")
    broadcast.UpdateWeightFromDistributed = MagicMock()
    monkeypatch.setitem(sys.modules, broadcast.__name__, broadcast)

    protocol = get_weight_transfer_protocol(_args())

    assert protocol is broadcast.UpdateWeightFromDistributed.return_value


def test_external_factory_rewords_a_missing_module_with_the_flag():
    with pytest.raises(ModuleNotFoundError) as exc_info:
        _load_external("definitely_missing_pkg.mod")

    message = str(exc_info.value)
    assert "--custom-weight-transfer-protocol-path" in message
    assert "'definitely_missing_pkg'" in message
    assert "Install the package that provides it or fix the module path." in message
    assert exc_info.value.name == "definitely_missing_pkg"


def test_external_factory_names_the_missing_parent_package():
    with pytest.raises(ModuleNotFoundError) as exc_info:
        _load_external("definitely_missing_pkg.sub.mod.build_protocol")

    message = str(exc_info.value)
    assert "--custom-weight-transfer-protocol-path" in message
    assert "'definitely_missing_pkg.sub.mod'" in message
    assert exc_info.value.name == "definitely_missing_pkg"


def test_external_factory_reraises_inner_import_failures_untouched(monkeypatch, tmp_path):
    (tmp_path / "mod_with_missing_inner_dep.py").write_text("import definitely_missing_inner_dep\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ModuleNotFoundError) as exc_info:
        _load_external("mod_with_missing_inner_dep.build_protocol")

    assert exc_info.value.name == "definitely_missing_inner_dep"
    assert "--custom-weight-transfer-protocol-path" not in str(exc_info.value)


def test_external_factory_describes_a_failing_module_getattr(monkeypatch, tmp_path):
    (tmp_path / "mod_with_raising_getattr.py").write_text(
        "def __getattr__(name):\n    raise AttributeError(f'module bug: lookup of {name!r} exploded')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(AttributeError) as exc_info:
        _load_external("mod_with_raising_getattr.build_protocol")

    message = str(exc_info.value)
    assert "--custom-weight-transfer-protocol-path" in message
    assert "'mod_with_raising_getattr.build_protocol'" in message
    assert "__getattr__" in message
    assert isinstance(exc_info.value.__cause__, AttributeError)
    assert "module bug: lookup of 'build_protocol' exploded" in str(exc_info.value.__cause__)


def test_external_factory_rewords_a_plain_missing_attribute(monkeypatch, tmp_path):
    (tmp_path / "mod_without_the_attribute.py").write_text("something_else = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(AttributeError) as exc_info:
        _load_external("mod_without_the_attribute.build_protocol")

    message = str(exc_info.value)
    assert "--custom-weight-transfer-protocol-path" in message
    assert "'mod_without_the_attribute'" in message
    assert "'build_protocol'" in message
    assert isinstance(exc_info.value.__cause__, AttributeError)


def test_external_factory_propagates_an_import_time_attribute_error_unchanged(monkeypatch, tmp_path):
    (tmp_path / "mod_raising_at_import.py").write_text("raise AttributeError('boom at import')\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(AttributeError) as exc_info:
        _load_external("mod_raising_at_import.build_protocol")

    assert str(exc_info.value) == "boom at import"


def test_external_factory_rejects_an_awaitable_result():
    async def _build(args):
        return None

    with (
        function_registry.temporary("test:awaiting_target", lambda args: _build(args)),
        pytest.raises(TypeError) as exc_info,
    ):
        _load_external("test:awaiting_target")

    message = str(exc_info.value)
    assert "'test:awaiting_target'" in message
    assert "returned an awaitable" in message
    assert "synchronous" in message
