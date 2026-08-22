"""Test stack for the tinker frontend: a real MultiLoraOperationBackend (registry +
ledger + validation) driven by a fake trainer loop. Only the Ray/trainer/GPU
boundary is faked — the fake driver speaks exactly the documented controller
verbs the Megatron driver uses (claim/commit/complete/retire/bootstrap/
mark_ready/record_weight_update), so ordering, dirty pins, fencing, and the
publish barrier behave like production."""

import asyncio
from types import SimpleNamespace

from miles.ray.multi_lora.backend import MultiLoraOperationBackend
from miles.ray.multi_lora.registry import AdapterState


def make_backend(router_url: str = "http://127.0.0.1:9", save_root: str = "/tmp/tinker-frontend-test", **overrides):
    args = SimpleNamespace(
        multi_lora_n_adapters=4,
        save=save_root,
        lora_rank=32,
        lora_alpha=64,
        hf_checkpoint="Qwen/Qwen3-0.6B",
        tinker_api_key=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return MultiLoraOperationBackend(args, router_url)


class FakeDriver:
    """The trainer/driver loop, minus the GPUs. Deterministic results:
    logprob rows are ``base - 0.01 * step`` so weights visibly "move" after
    an optim_step; named states are immutable; loads restore the step."""

    def __init__(self, backend: MultiLoraOperationBackend, base_logprob: float = -0.5) -> None:
        self.backend = backend
        self.base_logprob = base_logprob
        self.saved_states: dict[str, int] = {}
        self.paused = False
        # The fake driver IS the trainer: constructing it mirrors the real
        # driver flipping readiness once the training actors exist.
        backend.mark_trainer_ready()

    async def run(self, interval: float = 0.005) -> None:
        while True:
            if not self.paused:
                await self.tick()
            await asyncio.sleep(interval)

    async def tick(self) -> None:
        registry = self.backend.registry
        await self.backend.retire_adapters()
        for name in sorted(registry.in_state(AdapterState.CLEANUP)):
            await self.backend.free_slot(name)
        registry.bootstrap_pending()
        registry.mark_ready(
            [name for name, r in registry.in_state(AdapterState.PENDING).items() if r.slot is not None]
        )
        self._run_data_operations()
        self._run_control_operations()

    def _row(self, name: str, length: int) -> list[float]:
        step = self.backend.adapter_step(name)
        return [self.base_logprob - 0.01 * step] * length

    def _run_data_operations(self) -> None:
        for name, run in list(self.backend.registry.ready_adapters().items()):
            # Claim-and-bind, exactly like the rollout adapter's port.
            while (op := self.backend.claim_data_operation(name, run.registration_id)) is not None:
                rows = [self._row(name, sample["response_length"]) for sample in op["payload"]["samples"]]
                # Batch commits carry exact registration keys, never bare names.
                accumulated = [(name, run.registration_id)] if op["kind"] == "forward_backward" else []
                self.backend.commit_tinker_batch(accumulated, [op["operation_id"]], {op["operation_id"]: rows})

    def _run_control_operations(self) -> None:
        # Control claims return one envelope per batch: the operations plus a
        # BatchExecutionLease (the fake trainer has no local residency to
        # validate, and release is a no-op under fixed residency).
        claimed = self.backend.claim_ready_control_operations()
        for op in claimed["operations"]:
            kind, name, payload = op["kind"], op["name"], op.get("payload") or {}
            if kind == "optim_step":
                if op.get("poison"):
                    # Mirror the trainer: discard the poisoned window (no real
                    # grads here) and fail the step as a user error whose
                    # outcome confirms the window was physically consumed.
                    result = dict(ok=False, error=op["poison"], category="user", gradient_window_consumed=True)
                else:
                    adam = payload.get("adam_params") or {}
                    result = dict(ok=True, result=dict(grad_norm=0.125, learning_rate=adam.get("learning_rate", 1e-4)))
            elif kind == "save_state":
                tag = str(payload.get("tag") or f"step_{op['step']}")
                save_dir = self.backend.registry.find(name).config.save
                path = f"{save_dir}/{tag}"
                if path in self.saved_states:
                    result = dict(
                        ok=False, error=f"state '{tag}' already exists; states are immutable", category="user"
                    )
                else:
                    self.saved_states[path] = op["step"]
                    result = dict(ok=True, result=dict(path=path, step=op["step"]))
            elif kind == "load_state":
                path = payload.get("path")
                if path not in self.saved_states:
                    result = dict(ok=False, error=f"no state at '{path}'", category="user")
                else:
                    result = dict(ok=True, result=dict(step=self.saved_states[path], path=path))
            elif kind == "save_weights_for_sampler":
                # The publish barrier: the version bump lands BEFORE the
                # operation completes, like the driver's update_weights.
                self.backend.registry.record_weight_update([name])
                result = dict(ok=True)
            else:
                result = dict(ok=False, error=f"fake driver cannot run '{kind}'", category="server")
            self.backend.complete_control_operations({op["operation_id"]: result})
        if claimed["lease"] is not None:
            self.backend.release_batch_lease(claimed["lease"])


class FakeRouter:
    """Stands in for the sglang router's /generate contract (the shape the
    real frontend consumes): echoes deterministic tokens/logprobs and records
    every payload for assertions. Serves /get_server_info in the real
    response shape (ServerArgs echo + scheduler_info) so the frontend's
    context-limit discovery runs against it: ``context_length`` stays null —
    the launch-derived default — forcing the ``max_req_input_len + 6``
    reconstruction the scheduler math implies."""

    def __init__(self, max_req_input_len: int = 4090) -> None:
        self.requests: list[dict] = []
        self.max_req_input_len = max_req_input_len
        self.server_info_calls = 0

    def app(self):
        from fastapi import FastAPI, Request

        app = FastAPI()

        @app.post("/generate")
        async def generate(request: Request) -> dict:
            payload = await request.json()
            self.requests.append(payload)
            return self.response_for(payload)

        @app.get("/get_server_info")
        async def get_server_info() -> dict:
            self.server_info_calls += 1
            return {"context_length": None, "max_req_input_len": self.max_req_input_len, "status": "ready"}

        return app

    def response_for(self, payload: dict) -> dict:
        max_new = int((payload.get("sampling_params") or {}).get("max_new_tokens") or 4)
        n = min(max_new, 3)
        input_ids = payload.get("input_ids") or []
        meta_info = {
            "finish_reason": {"type": "length" if n == max_new else "stop"},
            "output_token_logprobs": [[-0.25 * (i + 1), 1000 + i, None] for i in range(n)],
            "prompt_tokens": len(input_ids),
        }
        if payload.get("logprob_start_len") == 0:
            # Real sglang shape: one entry per prompt token, first logprob None (no context).
            meta_info["input_token_logprobs"] = [
                [None if i == 0 else -0.125 * i, token, None] for i, token in enumerate(input_ids)
            ]
        return {"text": "ok", "meta_info": meta_info}
