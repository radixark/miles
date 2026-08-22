"""Miles rollout integrations for Polar-managed agent sessions.

This package wires the Polar rollout server into Miles' training loop. The
public surface re-exports the canonical entrypoints from the sibling modules so
a trainer (or the example launch scripts) can import them directly from
``miles.rollout`` without reaching into module internals:

- :mod:`.polar_config` — resolve Miles args into a :class:`PolarSlimeConfig`
  and render Polar task/topology payloads.
- :mod:`.polar_rollout` — the async rollout entrypoint
  ``generate_rollout_polar_async`` driven by ``--rollout-function-path``.
- :mod:`.polar_adapter` — convert Polar session results into Miles ``Sample``s.
- :mod:`.polar_reward` — the ``--custom-rm-path`` async entrypoint
  ``custom_rm`` and the leave-one-trajectory-out reward post-processor.
- :mod:`.polar_data_source` — ``CeilEpochRolloutDataSourceWithBuffer``, which
  rounds epoch length up to a whole number of rollout batches.

Polar core is an optional, runtime-only dependency; importing this package does
not require ``polar`` to be installed.
"""

from __future__ import annotations

# Public surface is resolved lazily via module ``__getattr__`` so ``import
# miles.rollout`` does not eagerly pull the heavy sibling chains (notably
# ``polar_data_source`` -> ``miles.rollout.data_source`` -> ``sglang``). This
# keeps the package importable under a plain Miles environment that has the
# thin polar bridge deps but not the full training stack.

_EXPORTS = {
    # polar_config
    "PolarSlimeConfig": (".polar_config", "PolarSlimeConfig"),
    "resolve_polar_slime_config": (".polar_config", "resolve_polar_slime_config"),
    "resolve_sglang_router_base_url": (".polar_config", "resolve_sglang_router_base_url"),
    "render_task_payload": (".polar_config", "render_task_payload"),
    "render_instruction": (".polar_config", "render_instruction"),
    "render_topology_template": (".polar_config", "render_topology_template"),
    # polar_rollout
    "generate_rollout_polar_async": (".polar_rollout", "generate_rollout_polar_async"),
    "AsyncPolarRolloutWorker": (".polar_rollout", "AsyncPolarRolloutWorker"),
    "get_global_async_worker": (".polar_rollout", "get_global_async_worker"),
    "stop_global_worker": (".polar_rollout", "stop_global_worker"),
    "PolarRolloutSchedulerError": (".polar_rollout", "PolarRolloutSchedulerError"),
    "PolarLowCompleteAcceptFractionError": (".polar_rollout", "PolarLowCompleteAcceptFractionError"),
    # polar_adapter
    "session_result_to_samples": (".polar_adapter", "session_result_to_samples"),
    "RolloutLogprobError": (".polar_adapter", "RolloutLogprobError"),
    # polar_reward
    "custom_rm": (".polar_reward", "custom_rm"),
    "compute_reward": (".polar_reward", "compute_reward"),
    "reward_func": (".polar_reward", "reward_func"),
    "post_process_rewards": (".polar_reward", "post_process_rewards"),
    # polar_data_source
    "CeilEpochRolloutDataSourceWithBuffer": (".polar_data_source", "CeilEpochRolloutDataSourceWithBuffer"),
    "ceil_to_batch_size": (".polar_data_source", "ceil_to_batch_size"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> object:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = target
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
