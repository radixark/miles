"""Shared fixtures for the backend trainer actor tests."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest


@pytest.fixture(scope="module")
def megatron_actor_module():
    """Import the Megatron trainer actor with its optional heavy dependencies stubbed.

    The peer-to-peer weight updater is replaced by a guard, so a test that reaches it
    fails loudly. Every stub and every module entry is restored afterwards.
    """
    pytest.importorskip("megatron")
    actor_module_name = "miles.backends.megatron_utils.actor"
    p2p_module_name = "miles.backends.megatron_utils.update_weight.update_weight_from_distributed.p2p"
    actor_package = importlib.import_module("miles.backends.megatron_utils")
    p2p_package = importlib.import_module("miles.backends.megatron_utils.update_weight.update_weight_from_distributed")
    missing = object()
    saved_actor_module = sys.modules.get(actor_module_name, missing)
    saved_p2p_module = sys.modules.get(p2p_module_name, missing)
    saved_saver = sys.modules.get("torch_memory_saver", missing)
    saved_actor_package_attr = getattr(actor_package, "actor", missing)
    saved_p2p_package_attr = getattr(p2p_package, "p2p", missing)

    saver_module = ModuleType("torch_memory_saver")
    saver_module.torch_memory_saver = Mock()
    p2p_module = ModuleType(p2p_module_name)
    p2p_module.UpdateWeightP2P = Mock(side_effect=AssertionError("these tests must not construct UpdateWeightP2P"))
    sys.modules["torch_memory_saver"] = saver_module
    sys.modules[p2p_module_name] = p2p_module
    p2p_package.p2p = p2p_module
    sys.modules.pop(actor_module_name, None)
    if saved_actor_package_attr is not missing:
        delattr(actor_package, "actor")

    try:
        yield importlib.import_module(actor_module_name)
    finally:
        sys.modules.pop(actor_module_name, None)
        if saved_actor_module is not missing:
            sys.modules[actor_module_name] = saved_actor_module
        if saved_actor_package_attr is missing:
            if hasattr(actor_package, "actor"):
                delattr(actor_package, "actor")
        else:
            actor_package.actor = saved_actor_package_attr
        sys.modules.pop(p2p_module_name, None)
        if saved_p2p_module is not missing:
            sys.modules[p2p_module_name] = saved_p2p_module
        if saved_p2p_package_attr is missing:
            if hasattr(p2p_package, "p2p"):
                delattr(p2p_package, "p2p")
        else:
            p2p_package.p2p = saved_p2p_package_attr
        if saved_saver is missing:
            sys.modules.pop("torch_memory_saver", None)
        else:
            sys.modules["torch_memory_saver"] = saved_saver
