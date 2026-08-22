from __future__ import annotations

from typing import NamedTuple

from miles.utils.workers import env_vars as worker_env_vars
from miles.utils.workers.worker_spec import BaseWorkerSpec

RENDERED_CELL_INDEX = 0

LEADER_ADDRESS_PLACEHOLDER = "$(LWS_LEADER_ADDRESS)"

WORKER_INDEX_SENTINEL = 987654321
_WORKER_INDEX_PLACEHOLDER = "$(LWS_WORKER_INDEX)"

_BASE_GPU_ID_SENTINEL = 987654322
_BASE_GPU_ID_PLACEHOLDER = f"$({worker_env_vars.BASE_GPU_ID_ENV_VAR})"


class _Substitution(NamedTuple):
    sentinel: int
    placeholder: str
    built_out_of: str


_SUBSTITUTIONS = (
    _Substitution(sentinel=WORKER_INDEX_SENTINEL, placeholder=_WORKER_INDEX_PLACEHOLDER, built_out_of="pod index"),
    _Substitution(sentinel=_BASE_GPU_ID_SENTINEL, placeholder=_BASE_GPU_ID_PLACEHOLDER, built_out_of="base gpu id"),
)


def real_or_sentinel_gpu_ids(spec: BaseWorkerSpec, *, is_sub_node: bool) -> list[int]:
    gpus_per_pod = max(1, spec.scheduling.gpus_per_pod())
    if is_sub_node:
        return [_BASE_GPU_ID_SENTINEL] * gpus_per_pod
    return list(range(gpus_per_pod))


def sentinels_to_placeholders(argv: list[str], spec: BaseWorkerSpec) -> list[str]:
    for substitution in _SUBSTITUTIONS:
        sentinel = str(substitution.sentinel)
        _assert_sentinel_is_a_whole_token(argv, sentinel=sentinel, spec=spec, built_out_of=substitution.built_out_of)
        argv = [substitution.placeholder if argument == sentinel else argument for argument in argv]
    return argv


def _assert_sentinel_is_a_whole_token(
    argv: list[str], *, sentinel: str, spec: BaseWorkerSpec, built_out_of: str
) -> None:
    embedded = [argument for argument in argv if sentinel in argument and argument != sentinel]
    assert not embedded, (
        f"Spec '{spec.name}' builds {embedded} out of its {built_out_of}; the value is substituted a whole "
        f"argument at a time, so it has to reach the command unchanged"
    )
