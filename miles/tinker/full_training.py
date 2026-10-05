"""Full checkpoints share one inference fleet, with snapshot switches serialized."""

import asyncio

from miles.tinker.runtime import MilesBackend


class FullTrainingBackend(MilesBackend):
    def __init__(self, trainer, router_url: str, *, inference_controller, base_checkpoint: str, dp_size: int = 1):
        super().__init__(trainer, router_url, dp_size, full_training=True)
        self.inference_controller = inference_controller
        self.base_checkpoint = base_checkpoint
        self._sampling_lock = asyncio.Lock()
        self._loaded_snapshot = None

    async def sample(self, payload: dict, lora_name: str | None, lora_path: str | None = None) -> dict:
        async with self._sampling_lock:
            # Cancellation before admission does not enqueue generation. After
            # admission, finish the calls before another sampler replaces weights.
            task = asyncio.create_task(self._sample_snapshot(payload, lora_name, lora_path))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                try:
                    await task
                finally:
                    raise

    async def _sample_snapshot(self, payload: dict, name: str | None, path: str | None) -> dict:
        try:
            await self.pin_snapshot(name, path)
            return await super().sample(payload, None)
        except Exception as error:
            # A failed serving replica does not invalidate the training model.
            return {"error": str(error)}

    async def pin_snapshot(self, name: str | None, path: str | None) -> None:
        info = await self.inference_controller.start_update_weights()
        complete = False
        try:
            if not info.rollout_engines:
                raise RuntimeError("no inference engines available for full-model sampling")
            version = name or "tinker-base"
            identity = (version, info.snapshot_cell_id_to_hashes)
            if identity != self._loaded_snapshot:
                self._loaded_snapshot = None
                results = await asyncio.gather(
                    *[
                        client.update_weights_from_disk(
                            model_path=path or self.base_checkpoint, load_format="auto", weight_version=version
                        )
                        for client in info.rollout_engines
                    ],
                    return_exceptions=True,
                )
                for result in results:
                    if isinstance(result, BaseException):
                        raise result
                    if isinstance(result, dict) and result.get("success") is False:
                        raise RuntimeError(f"inference weight load failed: {result}")
                versions = await asyncio.gather(*[client.get_weight_version() for client in info.rollout_engines])
                if not all(str(actual) == version for actual in versions):
                    raise RuntimeError(f"inference replicas did not load snapshot {version!r}: {versions}")
                self._loaded_snapshot = identity
            complete = True
        finally:
            if complete:
                await self.inference_controller.end_update_weights(info.snapshot_cell_id_to_hashes)
            else:
                self._loaded_snapshot = None
                await self.inference_controller.abort_update_weights()
