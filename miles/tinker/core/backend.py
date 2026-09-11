class ExecutorBackend:
    """Backend contract using datums and plain lists, without torch or trainer dependencies."""

    def trainer_dead(self) -> bool:
        """Whether the training workers are gone; the dispatch loop exits instead of containing."""
        return False

    async def load_slot(
        self, slot: int, rank: int, alpha: float, ckpt_path: str | None = None, load_optimizer: bool = True
    ) -> None:
        raise NotImplementedError

    async def unload_slot(self, slot: int) -> None:
        raise NotImplementedError

    async def forward_backward(
        self, batch_id: int, slot_datums: list, loss_fn: str, loss_fn_config: dict
    ) -> list[dict]:
        """slot_datums: slot-sorted [(slot, datum)]. Returns one
        {"loss": float, "logprobs": [float]} per datum, in order."""
        raise NotImplementedError

    async def forward_only(self, batch_id: int, slot_datums: list, loss_fn: str, loss_fn_config: dict) -> list[dict]:
        raise NotImplementedError

    async def optim_step(self, adam_params_by_slot: dict[int, dict]) -> dict[int, dict]:
        """-> per-slot outcome: {"grad_norm": x} stepped, {"skipped_nonfinite": 1.0}
        dropped non-finite grads, {"error": msg} failed."""
        raise NotImplementedError

    async def zero_grads(self, slot: int) -> None:
        """Drop the slot's accumulated gradients."""
        raise NotImplementedError

    async def save_slot(self, slot: int, path: str) -> None:
        raise NotImplementedError

    async def export_slot(self, slot: int, rank: int, alpha: float, path: str) -> None:
        """Write the slot's adapter as an engine-loadable dir."""
        raise NotImplementedError

    async def push_slot(
        self, slot: int, lora_name: str, rank: int, alpha: float, lora_path: str | None = None
    ) -> None:
        raise NotImplementedError

    async def sample(self, payload: dict, lora_name: str | None, lora_path: str | None = None) -> dict:
        """-> {"sequences": [{"tokens", "logprobs", "stop_reason"}],
        "prompt_logprobs"?, "topk_prompt_logprobs"?}"""
        raise NotImplementedError
