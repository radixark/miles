import json
from types import SimpleNamespace

import pytest
import serve_tinker
from tests.fast.fixtures.serve_tinker_fakes import tinker_startup

from miles.tinker.runtime import MilesBackend
from miles.utils import http_utils
from miles.utils.async_utils import with_disposer

_ = tinker_startup


class TestServe:
    async def test_observed_engines_initialize_a_client_that_can_sample(self, tinker_startup: SimpleNamespace) -> None:
        """A gateway starting with unknown topology can sample after its engines initialize."""
        args = tinker_startup.args
        assert args.inference_runtime_mut_state.engine_count == 0

        await with_disposer(serve_tinker.serve, args)

        assert tinker_startup.trainer.request.checkpoint_load.load == "base-model"
        assert not tinker_startup.trainer.request.checkpoint_load.resume_from_ckpt
        assert args.inference_runtime_mut_state.engine_count == 2
        assert http_utils._http_client is not None
        assert http_utils._client_concurrency == 24
        result = await MilesBackend(trainer=tinker_startup.trainer, router_url="http://router:30000").sample(
            payload={
                "prompt_tokens": [1, 2],
                "num_samples": 1,
                "sampling_params": {"max_tokens": 1},
                "prompt_logprobs": False,
                "topk_prompt_logprobs": 0,
            },
            lora_name=None,
        )

        assert result == {"sequences": [{"tokens": [42], "logprobs": [-0.25], "stop_reason": "length"}]}
        [request] = tinker_startup.requests
        assert request.method == "POST"
        assert str(request.url) == "http://router:30000/generate"
        assert json.loads(request.content)["input_ids"] == [1, 2]

    async def test_topology_query_failure_disposes_the_initialized_controller(
        self, tinker_startup: SimpleNamespace
    ) -> None:
        """A failed topology query still cleans up the controller that started its engines."""
        tinker_startup.controller.state_error = RuntimeError("topology unavailable")

        with pytest.raises(RuntimeError, match="topology unavailable"):
            await with_disposer(serve_tinker.serve, tinker_startup.args)

        assert tinker_startup.controller.disposed
        assert http_utils._http_client is None

    async def test_different_resolved_trainer_base_is_rejected_before_startup(
        self, tinker_startup: SimpleNamespace
    ) -> None:
        """A resolved trainer base differing from the engine base prevents gateway startup."""
        tinker_startup.args.ref_load = "other-model"

        with pytest.raises(AssertionError, match="same frozen HF base"):
            await with_disposer(serve_tinker.serve, tinker_startup.args)

        assert not tinker_startup.controller.initialized
        assert tinker_startup.trainer.request is None
