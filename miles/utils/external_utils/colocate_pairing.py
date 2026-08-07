from __future__ import annotations

import logging
from typing import Any, Protocol

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.colocate_matching import (
    PairingLayout,
    assert_colocate_supported,
    assert_layout_pairs,
    component_pod_name,
    target_trainer_pod,
)

__all__ = [
    "GATE_NAME",
    "PairingController",
    "PairingLayout",
    "PodIndices",
    "assert_colocate_supported",
    "assert_layout_pairs",
    "is_gated",
    "component_pod_name",
    "main",
    "parse_component_pod_name",
    "release_patch",
    "target_trainer_pod",
]

logger = logging.getLogger(__name__)

GATE_NAME = "miles.radixark.io/colocate-pairing"

_RESYNC_PERIOD_SECONDS = 300.0

_HOSTNAME_LABEL = "kubernetes.io/hostname"


def release_patch(*, node_name: str, has_node_selector: bool = False) -> list[dict[str, Any]]:
    pin = (
        {"op": "add", "path": f"/spec/nodeSelector/{_escape_pointer(_HOSTNAME_LABEL)}", "value": node_name}
        if has_node_selector
        else {"op": "add", "path": "/spec/nodeSelector", "value": {_HOSTNAME_LABEL: node_name}}
    )
    return [pin, {"op": "remove", "path": "/spec/schedulingGates"}]


def _escape_pointer(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def is_gated(pod: Any) -> bool:
    return any(gate.name == GATE_NAME for gate in (pod.spec.scheduling_gates or []))


class PairingController:
    def __init__(
        self,
        *,
        engine_component: str,
        trainer_component: str,
        layout: PairingLayout,
        pods: PairingPodApi,
    ) -> None:
        self._engine_component = engine_component
        self._trainer_component = trainer_component
        self._layout = layout
        self._pods = pods
        self._engine_of_trainer = {
            target_trainer_pod(
                engine_cell_index=cell_index,
                engine_pod_index=pod_index,
                layout=layout,
                trainer_component=trainer_component,
            ): component_pod_name(component=engine_component, cell_index=cell_index, pod_index=pod_index)
            for cell_index in range(layout.engine_cells)
            for pod_index in range(layout.pods_per_engine_cell)
        }
        assert len(self._engine_of_trainer) == layout.engine_cells * layout.pods_per_engine_cell, (
            f"two engine pods target the same trainer pod under {layout}, so one of them would never be "
            f"woken by the trainer it waits on"
        )

    async def reconcile(self, engine_pod_name: str) -> None:
        known = {pod.metadata.name: pod for pod in self._pods.pods_for(parent_key=engine_pod_name)}

        pod = known.get(engine_pod_name)
        if pod is None or not is_gated(pod):
            return

        indices = parse_component_pod_name(pod_name=engine_pod_name, component=self._engine_component)
        if indices is None:
            return

        target = target_trainer_pod(
            engine_cell_index=indices.cell_index,
            engine_pod_index=indices.pod_index,
            layout=self._layout,
            trainer_component=self._trainer_component,
        )
        trainer = known.get(target)
        node_name = trainer.spec.node_name if trainer is not None else None
        if not node_name:
            logger.info("waiting for %s to be scheduled before releasing %s", target, engine_pod_name)
            return

        logger.info("releasing %s onto %s, where %s runs", engine_pod_name, node_name, target)
        await self._pods.patch(
            pod_name=engine_pod_name,
            patch=release_patch(node_name=node_name, has_node_selector=bool(pod.spec.node_selector)),
        )

    def engine_waiting_on(self, trainer_pod_name: str) -> str | None:
        return self._engine_of_trainer.get(trainer_pod_name)

    def key_of(self, pod: Any) -> str:
        name = pod.metadata.name
        if parse_component_pod_name(pod_name=name, component=self._engine_component) is not None:
            return name
        if (waiting := self.engine_waiting_on(name)) is not None:
            return waiting
        return f"__unrelated__/{name}"


class PairingPodApi(Protocol):
    def pods_for(self, *, parent_key: str) -> list[Any]: ...

    async def patch(self, *, pod_name: str, patch: list[dict[str, Any]]) -> None: ...


class PodIndices(FrozenStrictBaseModel):
    cell_index: int
    pod_index: int


def parse_component_pod_name(*, pod_name: str, component: str) -> PodIndices | None:
    if not pod_name.startswith(f"{component}-"):
        return None

    remainder = pod_name[len(component) + 1 :].split("-")
    if not all(part.isdigit() for part in remainder) or not 1 <= len(remainder) <= 2:
        return None
    return PodIndices(cell_index=int(remainder[0]), pod_index=int(remainder[1]) if len(remainder) == 2 else 0)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--engine-component", required=True)
    parser.add_argument("--trainer-component", required=True)
    parser.add_argument("--engine-cells", type=int, required=True)
    parser.add_argument("--trainer-cells", type=int, required=True)
    parser.add_argument("--pods-per-engine-cell", type=int, required=True)
    parser.add_argument("--pods-per-trainer-cell", type=int, required=True)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    layout = PairingLayout(
        engine_cells=args.engine_cells,
        trainer_cells=args.trainer_cells,
        pods_per_engine_cell=args.pods_per_engine_cell,
        pods_per_trainer_cell=args.pods_per_trainer_cell,
    )
    assert_layout_pairs(layout=layout)
    asyncio.run(
        _run_forever(
            namespace=args.namespace,
            release=args.release,
            engine_component=args.engine_component,
            trainer_component=args.trainer_component,
            layout=layout,
        )
    )
    return 0


async def _run_forever(
    *, namespace: str, release: str, engine_component: str, trainer_component: str, layout: PairingLayout
) -> None:
    import asyncio

    from kubernetes_asyncio import client, config

    from miles.utils.workers.reconcile.k8s_api import KubernetesAsyncioPodApi
    from miles.utils.workers.reconcile.k8s_reflector import KubernetesReflector
    from miles.utils.workers.reconcile.loop import ReconcileLoop

    try:
        config.load_incluster_config()
    except config.ConfigException:
        await config.load_kube_config()

    async with client.ApiClient() as api_client:
        core_v1 = client.CoreV1Api(api_client)
        pods = StorePodApi(core_v1=core_v1, namespace=namespace)
        controller = PairingController(
            engine_component=engine_component,
            trainer_component=trainer_component,
            layout=layout,
            pods=pods,
        )
        reflector = KubernetesReflector(
            kube_client=KubernetesAsyncioPodApi(core_v1_api=core_v1),
            namespace=namespace,
            label_selector=f"app.kubernetes.io/instance={release}",
        )
        loop = ReconcileLoop(
            source=reflector.watch,
            reconcile=controller.reconcile,
            key_map=controller.key_of,
            resync_period=_RESYNC_PERIOD_SECONDS,
        )
        pods.read_from(loop=loop)
        await loop.start()
        logger.info("pairing controller watching %s in %s", release, namespace)
        try:
            await asyncio.Event().wait()
        finally:
            await loop.stop()


class StorePodApi:
    def __init__(self, *, core_v1: Any, namespace: str) -> None:
        self._core_v1 = core_v1
        self._namespace = namespace
        self._loop: Any = None

    def read_from(self, *, loop: Any) -> None:
        self._loop = loop

    def pods_for(self, *, parent_key: str) -> list[Any]:
        assert self._loop is not None, "the reconcile loop has to be attached before pods are read"
        return self._loop.get_by_parent(parent_key)

    async def patch(self, *, pod_name: str, patch: list[dict[str, Any]]) -> None:
        await self._core_v1.patch_namespaced_pod(name=pod_name, namespace=self._namespace, body=patch)


if __name__ == "__main__":
    raise SystemExit(main())
