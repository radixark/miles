"""Node-local defaults for the JIT compile caches (#3158).

Triton, TorchInductor, tvm-ffi and SGLang default their caches to the home
directory, which is NFS on most clusters. Many ranks on several nodes
cold-compiling the same kernels into one shared cache race for it, and the
first training step after a rollout then looks like a hang. The same literal
path on every node keeps each node's cache local; it is namespaced by user so
two jobs sharing a node never fight over ownership of one directory.

``import sglang`` calls ``os.environ.setdefault`` for the Triton and Inductor
variables under ``SGLANG_CACHE_DIR`` (default ``~/.cache/sglang``), so a
process that imported sglang, such as the training driver, already carries
those two keys. They count as user intent only when ``SGLANG_CACHE_DIR`` is
set; otherwise they are sglang's own defaults and get replaced like an unset
value.
"""

import getpass
import os
from collections.abc import Mapping

NODE_LOCAL_COMPILE_CACHE_ENV_VARS = {
    "SGLANG_CACHE_DIR": "sglang",
    "TRITON_CACHE_DIR": "triton",
    "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
    "TVM_FFI_CACHE_DIR": "tvm-ffi",
}
# The layout sglang.srt.environ.third_party_cache_defaults() derives from SGLANG_CACHE_DIR.
_SGLANG_DERIVED = {"TRITON_CACHE_DIR": "triton", "TORCHINDUCTOR_CACHE_DIR": "inductor"}
_SGLANG_DEFAULT_BASE = "~/.cache/sglang"


def _user() -> str:
    try:
        return getpass.getuser()
    except (KeyError, OSError):
        return str(os.getuid())


def node_local_compile_cache_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Cache env vars for a worker process. A value the caller set wins; an
    empty value, or one sglang's import-time setdefault produced, counts as
    unset. With a user-set ``SGLANG_CACHE_DIR`` the Triton and Inductor caches
    follow sglang's layout under it."""
    if environ is None:
        environ = os.environ
    root = f"/tmp/miles-compile-cache-{_user()}"
    sglang_base = environ.get("SGLANG_CACHE_DIR") or None
    sglang_default_base = os.path.expanduser(_SGLANG_DEFAULT_BASE)

    env: dict[str, str] = {}
    for key, subdir in NODE_LOCAL_COMPILE_CACHE_ENV_VARS.items():
        value = environ.get(key) or None
        if key in _SGLANG_DERIVED:
            if sglang_base is None and value == os.path.join(sglang_default_base, _SGLANG_DERIVED[key]):
                value = None  # sglang's own default, not the user's choice
            default = os.path.join(sglang_base, _SGLANG_DERIVED[key]) if sglang_base else f"{root}/{subdir}"
        else:
            default = f"{root}/{subdir}"
        env[key] = value or default
    return env
