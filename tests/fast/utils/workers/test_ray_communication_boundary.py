from __future__ import annotations

import functools
from pathlib import Path

from tests.fast.source_scan import (
    FRAMEWORK_ROOT,
    REPO_ROOT,
    imported_modules,
    imported_modules_of_source,
    imports_package,
    relative_paths,
    shipped_modules,
)

EXCLUDED_DIRS = (REPO_ROOT / "tests",)

RAY_USING_MODULES = {
    "miles/ray/placement_group.py": "launcher closure: placement groups are how ray is asked to schedule",
    "miles/ray/wiring.py": "launcher closure: driver shutdown kills the manager it launched by ActorHandle",
    "miles/utils/ray_utils.py": "launcher closure: node lookup and pinning options for the launcher's own calls",
    "miles/utils/orchestration_utils.py": (
        "launcher closure: the shared driver composition root returns its ray worker manager"
    ),
    "miles/utils/workers/ray_worker_manager.py": "launcher closure: it is the launcher",
    "miles/utils/workers/ray_worker_handle.py": "launcher closure: the handle of the ray communication mode itself",
    "miles/utils/workers/worker_provider/ray.py": "launcher closure: it reads the launcher's own bookkeeping",
    "miles/utils/workers/backend_capability/ray.py": "launcher closure: it builds the ray provider and operations",
    "miles/utils/workers/cell_operations/ray.py": "launcher closure: suspend and resume are launcher verbs",
    "miles/utils/workers/addr_allocator.py": "launcher closure: ports are probed on the actor the launcher made",
    "miles/utils/external_utils/command_utils/ray_backend/command.py": "launcher closure: running the ray launch scripts on every node the launcher has",
    "miles/utils/misc.py": "launcher closure: the node probe every launched actor answers with",
    "miles/utils/http_utils.py": "launcher closure: reaching a port on a node the launcher scheduled",
    "miles/utils/object_store.py": "object store exemption: the ray object store is a data plane of its own",
    "miles/dashboard/backend.py": "known debt: the dashboard collector is a named actor, tracked outside M23",
    "miles/dashboard/collector.py": "known debt: the dashboard collector is a named actor, tracked outside M23",
    "miles/dashboard/hooks.py": "known debt: the dashboard reads its gpu ids from ray, tracked outside M23",
    "miles/utils/tracking_utils/prometheus_utils.py": "known debt: the prometheus collector is a ray actor, skipped under kubernetes",
    "miles/ray/train_actor.py": "launcher closure: a launched actor reads the gpu ids ray gave it",
    "miles/backends/fsdp_utils/update_weight_utils.py": "node ip lookup for a collective, not a call to another worker",
    "miles/backends/megatron_utils/update_weight/update_weight_from_distributed/broadcast.py": "node ip lookup for a collective, not a call to another worker",
    "miles/backends/megatron_utils/update_weight/update_weight_from_distributed/p2p_transfer_utils.py": "node ip lookup for a collective, not a call to another worker",
    "miles/utils/debug_utils/replay_reward_fn.py": "tooling: a standalone debugging script",
    "miles/utils/test_utils/mock_sglang_engine.py": "tooling: a test double that stands in for a ray-launched engine",
    "tools/convert_torch_dist_to_hf_ray.py": "tooling: a standalone conversion script that fans out over a ray cluster",
    "examples/experimental/formal_math/single_round/kimina_wrapper.py": "user example: a verifier pool of its own, outside the worker layer",
    "examples/experimental/formal_math/single_round/reward_fn.py": "user example: a verifier pool of its own, outside the worker layer",
}


@functools.cache
def _scanned_modules() -> tuple[Path, ...]:
    return tuple(shipped_modules(exclude_dirs=EXCLUDED_DIRS))


@functools.cache
def _imports_ray(path: Path) -> bool:
    return imports_package(imported_modules(path), "ray")


@functools.cache
def _ray_using_module_paths() -> tuple[str, ...]:
    return tuple(relative_paths(path for path in _scanned_modules() if _imports_ray(path)))


class TestRayIsOnlyUsedInsideTheLauncherClosure:
    def test_no_module_reaches_for_ray_without_being_listed(self):
        """Ray communication outside the launcher closure is what the rpc comm backend exists to remove."""
        unlisted = sorted(set(_ray_using_module_paths()) - set(RAY_USING_MODULES))

        assert unlisted == [], (
            f"{unlisted} import ray; a worker must be reachable over rpc too, so either drop the import or "
            f"add it to RAY_USING_MODULES with the reason it belongs to the launcher closure"
        )

    def test_the_list_names_no_module_that_stopped_using_ray(self):
        """A stale exemption reads as permission to bring ray back into a module that no longer needs it."""
        stale = sorted(set(RAY_USING_MODULES) - set(_ray_using_module_paths()))

        assert stale == []

    def test_every_listed_module_says_why(self):
        """The exemptions are a ledger of remaining debt, and a blank reason hides an entry from review."""
        assert sorted(name for name, reason in RAY_USING_MODULES.items() if not reason.strip()) == []

    def test_the_rpc_layer_itself_never_touches_ray(self):
        """The rpc client and server must run in a process that has no ray at all, such as a kubernetes pod."""
        rpc_modules = [path for path in _scanned_modules() if "rpc" in path.parts and _imports_ray(path)]

        assert rpc_modules == []

    def test_a_served_worker_is_started_without_ray(self):
        """serve_actor happens to run inside a ray actor, but it starts the same server a pod starts."""
        assert not _imports_ray(FRAMEWORK_ROOT / "utils" / "workers" / "serving" / "serve_actor.py")


class TestTheScanReachesEveryProcessTheDriverIsPartOf:
    def test_the_orchestration_scripts_are_scanned(self):
        """The driver is where a ray-only exception handler hid, and miles/ alone never covered it."""
        scanned = relative_paths(_scanned_modules())

        assert {"train.py", "train_async.py", "train_multi_lora_async.py"} <= set(scanned)

    def test_a_driver_script_that_reaches_for_ray_would_be_reported(self):
        """A check that only ever looks under miles/ passes on the very file that broke under rpc."""
        assert imports_package(imported_modules_of_source("import ray\n"), "ray")


class TestTheShapesThatUsedToSlipThrough:
    def test_a_submodule_import_counts(self):
        """`import ray.exceptions` reaches ray just as much as `import ray` does."""
        assert imports_package(imported_modules_of_source("import ray.exceptions\n"), "ray")

    def test_an_aliased_import_counts(self):
        """Renaming the module on the way in does not rename what it talks to."""
        assert imports_package(imported_modules_of_source("import ray as r\n"), "ray")

    def test_a_dynamic_import_counts(self):
        """importlib is the shape an import lands in once someone wants it to not look like one."""
        assert imports_package(imported_modules_of_source('import importlib\nimportlib.import_module("ray")\n'), "ray")

    def test_a_module_that_merely_shares_the_prefix_does_not_count(self):
        """A false positive costs the ledger its meaning, and `raydium` is not ray."""
        assert not imports_package(imported_modules_of_source("import raydium\n"), "ray")
