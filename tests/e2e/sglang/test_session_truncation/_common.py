"""Per-model session TITO runs with complete and with cut replies, on one SGLang server per model.

Each model file starts its engine once and runs four cases: session server v1
and v2, each with the multi-role driver as is (``complete``) and with replies cut
inside reasoning, a tool call or the answer (``truncate``). The driver, its gates
and the model settings are the multi-role suite's; this harness runs the
session servers in-process instead of through a Ray rollout, so one engine
serves every case.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from huggingface_hub import snapshot_download
from tests.e2e.sglang.test_session_server_multi_role._common import ModelConfig
from tests.e2e.sglang.utils.sglang_server import start_sglang_server
from tests.fast.fixtures.session_fixtures import make_session_server_config

from miles.rollout.base_types import GenerateFnInput
from miles.rollout.session.server import SessionServer
from miles.rollout.session.types import SessionServerInstance
from miles.utils import http_utils
from miles.utils.chat_template_utils import resolve_fixed_chat_template, resolve_reasoning_and_tool_call_parser
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizerType
from miles.utils.http_utils import find_available_port
from miles.utils.test_utils import session_verify_agent
from miles.utils.test_utils.session_truncation import CUT_KINDS
from miles.utils.test_utils.session_verify_runner import (
    SESSION_VERIFY_INVARIANT_ARGS,
    SPECULATIVE_SERVER_ARGS,
    assert_session_verify_metrics,
)
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer
from miles.utils.types import Sample

# (session server version, cut replies); the ids select cases with ``-k truncate`` / ``-k complete``.
CASES = [("v1", False), ("v1", True), ("v2", False), ("v2", True)]
CASE_IDS = ["v1-complete", "v1-truncate", "v2-complete", "v2-truncate"]

_PROMPTS_PER_CASE = 4
_SAMPLING_SEED = 20261011
_PICKER_PATH = "miles.rollout.session.v2.picker_hub.drop_same_prompt_retries"
_POSTPROCESSOR_PATH = "miles.rollout.session.v2.postprocessor_hub.default_postprocess"


@contextmanager
def serve_model(cfg: ModelConfig):
    """Start SGLang for *cfg* with its multi-role engine settings; yields ``(server, local model path)``.

    SGLang gets the family's fixed chat template, as Miles passes it, so its
    reasoning parser knows whether the generation prompt opens reasoning.
    """
    model_path = snapshot_download(cfg.model_name)
    reasoning_parser, tool_call_parser = resolve_reasoning_and_tool_call_parser(
        cfg.tito_model, cfg.reasoning_parser, cfg.tool_call_parser
    )
    chat_template_path, _ = resolve_fixed_chat_template(cfg.tito_model)
    server_args = {
        "chat-template": chat_template_path,
        "tp-size": cfg.tp_size,
        "ep-size": cfg.ep_size if cfg.ep_size > 1 else None,
        "reasoning-parser": reasoning_parser,
        "tool-call-parser": tool_call_parser,
        "context-length": cfg.context_length,
        "kv-cache-dtype": cfg.kv_cache_dtype,
        "mamba-full-memory-ratio": cfg.mamba_full_memory_ratio,
        "cuda-graph-backend-prefill": cfg.cuda_graph_backend_prefill,
        **(SPECULATIVE_SERVER_ARGS if cfg.enable_spec else {}),
    }
    extra_args = [arg for name, value in server_args.items() if value is not None for arg in (f"--{name}", str(value))]
    server = start_sglang_server(model_path=model_path, enable_deterministic_inference=False, extra_args=extra_args)
    try:
        yield server, model_path
    finally:
        server.stop()


def run_case(server, model_path: str, cfg: ModelConfig, *, version: str, truncate: bool) -> None:
    """Run one case's sessions against *server* and apply the multi-role gates."""
    with _serve_sessions(backend_url=server.base_url, model_path=model_path, cfg=cfg, version=version) as args:
        if (
            truncate
            and TITOTokenizerType.get_tokenizer_class(TITOTokenizerType(cfg.tito_model)).TRUNCATION_GRAMMAR is None
        ):
            asyncio.run(_assert_cut_turn_rejected(args))
            return
        args.session_verify_truncate = truncate
        metrics_fd, metrics_path = tempfile.mkstemp(prefix="session_truncation_metrics_", suffix=".jsonl")
        os.close(metrics_fd)
        with patch.dict(os.environ, {"MILES_SESSION_VERIFY_METRICS_PATH": metrics_path}):
            errors = asyncio.run(_generate(args, cfg))
        try:
            assert_session_verify_metrics(
                metrics_path,
                assistant_text_threshold=cfg.assistant_text_threshold,
                special_token_count_threshold=cfg.special_token_count_threshold,
                required_cut_kinds=CUT_KINDS if truncate else (),
            )
        except AssertionError as exc:
            exc.add_note(f"per-sample metrics kept at {metrics_path}")
            raise
        os.unlink(metrics_path)
    if errors:
        raise errors[0]


@contextmanager
def _serve_sessions(*, backend_url: str, model_path: str, cfg: ModelConfig, version: str):
    # Miles resolves the family's fixed template and kwargs at startup and hands both to the session server.
    chat_template_path, chat_template_kwargs = resolve_fixed_chat_template(cfg.tito_model)
    port = find_available_port(31000)
    instance_id = uuid.uuid4().hex
    args = SimpleNamespace(
        miles_router_timeout=600,
        hf_checkpoint=model_path,
        chat_template_path=chat_template_path,
        apply_chat_template_kwargs=chat_template_kwargs,
        tito_model=cfg.tito_model,
        sglang_speculative_algorithm=None,
        use_session_server=version,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        use_sampling_support_replay=False,
        pause_generation_mode="retract",
        session_server_ip="127.0.0.1",
        session_server_instances=[SessionServerInstance(addr=f"127.0.0.1:{port}", instance_id=instance_id)],
        save_debug_trajectory_data=None,
        custom_agent_function_path="miles.utils.test_utils.session_verify_agent.run_agent",
        custom_agent_function_mode="inline",
        partial_rollout=False,
        max_seq_len=None,
        session_verify_cycles=cfg.cycles,
        tool_call_failure_mode=cfg.tool_call_failure_mode,
        session_verify_truncate=False,
        session_sample_picker_path=_PICKER_PATH,
        session_sample_postprocessor_path=_POSTPROCESSOR_PATH,
    )
    config = make_session_server_config(
        port=port,
        instance_id=instance_id,
        backend_url=backend_url,
        timeout=args.miles_router_timeout,
        hf_checkpoint=model_path,
        chat_template_path=chat_template_path,
        apply_chat_template_kwargs=chat_template_kwargs,
        tito_model=cfg.tito_model,
        use_session_server=version,
        pause_generation_mode=args.pause_generation_mode,
        session_sample_picker_path=_PICKER_PATH,
        session_sample_postprocessor_path=_POSTPROCESSOR_PATH,
    )
    server = UvicornThreadServer(SessionServer(config).app, host=args.session_server_ip, port=port)
    server.start()
    try:
        yield args
    finally:
        server.stop()


async def _generate(args: SimpleNamespace, cfg: ModelConfig) -> list[BaseException]:
    """Run every sample through the multi-role generate wrapper; return the errors the gates do not cover."""
    sampling_params = {
        "model": args.hf_checkpoint,
        "temperature": SESSION_VERIFY_INVARIANT_ARGS["rollout_temperature"],
        "max_new_tokens": cfg.rollout_max_response_len,
    }
    inputs = [
        GenerateFnInput(
            state=SimpleNamespace(args=args),
            sample=Sample(index=index, rollout_id=index, prompt=session_verify_agent.build_initial_messages()),
            sampling_params={**sampling_params, "sampling_seed": _SAMPLING_SEED + index},
            evaluation=False,
        )
        for index in range(_PROMPTS_PER_CASE * cfg.n_samples_per_prompt)
    ]
    async with httpx.AsyncClient(timeout=None) as client:
        with patch.object(http_utils, "_http_client", client):
            results = await asyncio.gather(
                *(session_verify_agent.generate(generate_input) for generate_input in inputs), return_exceptions=True
            )
    # Assertion failures are journaled to the metrics file and reported by its gate.
    return [
        result for result in results if isinstance(result, BaseException) and not isinstance(result, AssertionError)
    ]


async def _assert_cut_turn_rejected(args: SimpleNamespace) -> None:
    """A family without a truncation grammar answers 409 when a cut turn is continued."""
    url = f"http://{args.session_server_instances[0].addr}"
    messages = session_verify_agent.build_initial_messages()
    async with httpx.AsyncClient(timeout=600) as client:
        response = await client.post(f"{url}/sessions", json={"temperature": 0.7})
        session_url = f"{url}/sessions/{response.json()['session_id']}"
        body = {"messages": messages, "tools": session_verify_agent.TOOLS, "max_tokens": 8}
        response = await client.post(f"{session_url}/v1/chat/completions", json=body)
        assert response.status_code == 200, response.text
        choice = response.json()["choices"][0]
        assert choice["finish_reason"] == "length", choice
        continued = messages + [
            choice["message"],
            {"role": "user", "content": session_verify_agent.USER_FOLLOWUP_TEXT},
        ]
        response = await client.post(f"{session_url}/v1/chat/completions", json={**body, "messages": continued})
        assert response.status_code == 409, response.text
