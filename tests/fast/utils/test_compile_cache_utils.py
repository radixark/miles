from __future__ import annotations

import os

import pytest

from miles.utils import compile_cache_utils
from miles.utils.compile_cache_utils import node_local_compile_cache_env

_REAL_USER_LOOKUP = compile_cache_utils._user


@pytest.fixture(autouse=True)
def _alice(monkeypatch):
    monkeypatch.setattr(compile_cache_utils, "_user", lambda: "alice")


def test_defaults_every_cache_to_a_node_local_per_user_path():
    """~/.triton, ~/.cache/torchinductor, ~/.cache/tvm-ffi and ~/.cache/sglang are NFS on clusters (#3158)."""
    assert node_local_compile_cache_env(environ={}) == {
        "SGLANG_CACHE_DIR": "/tmp/miles-compile-cache-alice/sglang",
        "TRITON_CACHE_DIR": "/tmp/miles-compile-cache-alice/triton",
        "TORCHINDUCTOR_CACHE_DIR": "/tmp/miles-compile-cache-alice/torchinductor",
        "TVM_FFI_CACHE_DIR": "/tmp/miles-compile-cache-alice/tvm-ffi",
    }


def test_the_callers_own_value_wins_per_variable():
    env = node_local_compile_cache_env(environ={"TRITON_CACHE_DIR": "/nvme/triton", "TVM_FFI_CACHE_DIR": ""})

    assert env["TRITON_CACHE_DIR"] == "/nvme/triton"
    assert env["TORCHINDUCTOR_CACHE_DIR"] == "/tmp/miles-compile-cache-alice/torchinductor"
    # An empty value counts as unset.
    assert env["TVM_FFI_CACHE_DIR"] == "/tmp/miles-compile-cache-alice/tvm-ffi"


def test_sglangs_import_time_defaults_do_not_count_as_user_intent():
    """`import sglang` setdefaults the Triton and Inductor dirs under ~/.cache/sglang in the driver."""
    home = os.path.expanduser("~/.cache/sglang")
    env = node_local_compile_cache_env(
        environ={"TRITON_CACHE_DIR": f"{home}/triton", "TORCHINDUCTOR_CACHE_DIR": f"{home}/inductor"}
    )

    assert env["TRITON_CACHE_DIR"] == "/tmp/miles-compile-cache-alice/triton"
    assert env["TORCHINDUCTOR_CACHE_DIR"] == "/tmp/miles-compile-cache-alice/torchinductor"


def test_a_user_set_sglang_cache_dir_owns_the_triton_and_inductor_caches():
    env = node_local_compile_cache_env(
        environ={"SGLANG_CACHE_DIR": "/nvme/sgl", "TRITON_CACHE_DIR": "/nvme/sgl/triton"}
    )

    assert env["SGLANG_CACHE_DIR"] == "/nvme/sgl"
    assert env["TRITON_CACHE_DIR"] == "/nvme/sgl/triton"
    assert env["TORCHINDUCTOR_CACHE_DIR"] == "/nvme/sgl/inductor"
    assert env["TVM_FFI_CACHE_DIR"] == "/tmp/miles-compile-cache-alice/tvm-ffi"


def test_falls_back_to_the_uid_when_the_user_name_is_unknown(monkeypatch):
    def no_name():
        raise KeyError("getpwuid(): uid not found")

    monkeypatch.setattr(compile_cache_utils, "_user", _REAL_USER_LOOKUP)
    monkeypatch.setattr(compile_cache_utils.getpass, "getuser", no_name)
    monkeypatch.setattr(compile_cache_utils.os, "getuid", lambda: 4242)

    assert node_local_compile_cache_env(environ={})["TRITON_CACHE_DIR"] == "/tmp/miles-compile-cache-4242/triton"
