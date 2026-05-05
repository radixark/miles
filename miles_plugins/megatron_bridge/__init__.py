"""Miles plugin package for ``megatron.bridge`` integration.

Importing this package is enough to:

* register miles' bridge subclasses (e.g.
  :class:`~miles_plugins.megatron_bridge.nemotron_h.MilesNemotronHBridge`) via
  ``@MegatronModelBridge.register_bridge`` so ``AutoBridge`` picks them up
  instead of the upstream defaults;
* install general-purpose shims that make ``megatron.bridge`` cooperate with
  miles infrastructure (e.g. ``ReloadableProcessGroup``).

Every shim / registration is wrapped in try/except so an import-time failure of
one model does not prevent other models from working.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _install_bridge_pp_group_unwrap() -> None:
    """Let ``MegatronParamMapping.broadcast_obj_from_pp_rank`` work with
    miles' :class:`~miles.utils.reloadable_process_group.ReloadableProcessGroup`.

    ``broadcast_obj_from_pp_rank`` calls ``broadcast_object_list`` on
    ``self.pp_group``, which goes through ``_world.pg_group_ranks``. Miles wraps
    every ``ProcessGroup`` in ``ReloadableProcessGroup`` for reload-safety; that
    wrapper is not in ``pg_group_ranks`` so ``get_group_rank`` raises
    ``"Group ... is not registered"``. Temporarily swap in the inner group for
    the duration of the broadcast.
    """
    from megatron.bridge.models.conversion.param_mapping import MegatronParamMapping

    from miles.utils.reloadable_process_group import ReloadableProcessGroup

    if getattr(MegatronParamMapping, "_miles_pp_group_unwrap_installed", False):
        return

    _orig = MegatronParamMapping.broadcast_obj_from_pp_rank

    def broadcast_obj_from_pp_rank(self, obj, name=None):
        if not isinstance(self.pp_group, ReloadableProcessGroup):
            return _orig(self, obj, name)
        saved = self.pp_group
        self.pp_group = saved.group
        try:
            return _orig(self, obj, name)
        finally:
            self.pp_group = saved

    MegatronParamMapping.broadcast_obj_from_pp_rank = broadcast_obj_from_pp_rank
    MegatronParamMapping._miles_pp_group_unwrap_installed = True


try:
    _install_bridge_pp_group_unwrap()
except Exception as _e:  # best-effort
    logger.warning("miles bridge shim _install_bridge_pp_group_unwrap not applied: %s", _e)


# Model-specific bridge subclasses. Each submodule self-installs on import.
# Keep imports here so merely importing ``miles_plugins.megatron_bridge`` is
# enough to pick up every miles bridge (mirrors ``miles_plugins.mbridge``).
try:
    from . import nemotron_h  # noqa: F401
except Exception as _e:  # pragma: no cover - defensive
    logger.warning("miles nemotron_h plugin failed to load: %s", _e)
