from collections.abc import Sequence

from miles.utils.args.component_rollout import InferenceRuntimeImmutState


class _SnapshotController:
    def __init__(self, snapshots: Sequence[InferenceRuntimeImmutState]) -> None:
        self._snapshots = iter(snapshots)

    async def get_inference_runtime_immut_state(self) -> InferenceRuntimeImmutState:
        return next(self._snapshots)


class _ControllerProvider:
    def __init__(self, controller: _SnapshotController) -> None:
        self._controller = controller

    def get_handle(self, name: str) -> _SnapshotController:
        return self._controller
