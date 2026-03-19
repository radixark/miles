import json
import logging
import time
from typing import TYPE_CHECKING

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from miles.router.session.session_types import GetSessionResponse, SessionRecord
from miles.router.session.single_user_turn_trajectory import SingleUserTurnTrajectoryManager
from miles.utils.chat_template_utils import get_tito_tokenizer
from miles.utils.processing_utils import load_tokenizer

if TYPE_CHECKING:
    from miles.router.router import MilesRouter

logger = logging.getLogger(__name__)


def setup_session_routes(app, router: "MilesRouter"):
    hf_checkpoint = getattr(router.args, "hf_checkpoint", None)
    if not hf_checkpoint:
        if getattr(router, "verbose", False):
            logger.info("[miles-router] Skipping session routes (hf_checkpoint not set).")
        return

    tokenizer = load_tokenizer(
        hf_checkpoint, chat_template_path=router.args.chat_template_path, trust_remote_code=True
    )

    tito_tokenizer = get_tito_tokenizer(
        tokenizer,
        tokenizer_type=getattr(router.args, "tito_model", "default"),
    )

    manager = SingleUserTurnTrajectoryManager(router.args, tokenizer, tito_tokenizer=tito_tokenizer)

    @app.post("/sessions")
    async def create_session():
        session_id = manager.create_session()
        return {"session_id": session_id}

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str):
        records = manager.get_session_records_by_id(session_id)
        if records is None:
            return JSONResponse(status_code=404, content={"error": "session not found"})
        metadata = manager.compute_session_metadata(session_id)
        metadata["accumulated_token_ids"] = manager.get_session_token_ids(session_id)
        metadata["max_trim_tokens"] = manager._tito_tokenizer.max_trim_tokens if manager._tito_tokenizer else 0
        return GetSessionResponse(
            session_id=session_id,
            records=records,
            metadata=metadata,
        )

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str):
        deleted = manager.delete_session_by_id(session_id)
        if deleted is None:
            return JSONResponse(status_code=404, content={"error": "session not found"})
        return Response(status_code=204)

    @app.post("/sessions/{session_id}/v1/chat/completions")
    async def chat_completions(request: Request, session_id: str):
        body = await request.body()
        request_body = json.loads(body) if body else {}

        request_body["logprobs"] = True
        request_body["return_prompt_token_ids"] = True
        request_body["return_meta_info"] = True
        if getattr(router.args, "use_rollout_routing_replay", False):
            request_body["return_routed_experts"] = True
        request_body["no_stop_trim"] = False

        request_messages = request_body.get("messages", [])
        pretokenized = manager.try_prepare_pretokenized(session_id, request_messages, tools=request_body.get("tools"))
        if pretokenized is not None:
            request_body["input_ids"] = pretokenized["input_ids"]
            logger.debug(
                "Using pretokenized input_ids: %d tokens",
                len(pretokenized["input_ids"]),
            )

        body = json.dumps(request_body).encode()

        result = await router._do_proxy(request, "v1/chat/completions", body=body)

        if result["status_code"] != 200:
            return router._build_proxy_response(result)

        response = json.loads(result["response_body"])

        choice = response.get("choices", [{}])[0]

        if "meta_info" not in choice or "output_token_logprobs" not in choice.get("meta_info", {}):
            raise RuntimeError("meta_info and output_token_logprobs must be in choice (requires logprobs=True)")

        assistant_message = choice.get("message", {})
        assert (
            assistant_message.get("content") is not None
        ), "assistant message content is None, when tool call parser failed SGLang should still return a empty content rather than None. Please check your modified SGLang version."

        prompt_token_ids = choice.get("prompt_token_ids")
        meta_info = choice["meta_info"]
        output_token_logprobs = meta_info["output_token_logprobs"]
        completion_tokens = meta_info["completion_tokens"]

        actual_output_logprobs_len = len(output_token_logprobs)
        if actual_output_logprobs_len != completion_tokens:
            raise RuntimeError(
                "invalid chat completion response: "
                f"len(output_token_logprobs)={actual_output_logprobs_len} "
                f"!= completion_tokens={completion_tokens}"
                f"Please check whether you use the correct SGLang branch which has fix the tokenizer batch decode issue."
            )

        completion_token_ids = [t[1] for t in output_token_logprobs]

        manager.update_pretokenized_state(
            session_id,
            request_messages,
            assistant_message,
            prompt_token_ids=prompt_token_ids,
            completion_token_ids=completion_token_ids,
        )

        record = SessionRecord(
            timestamp=time.time(),
            method=request.method,
            path="/v1/chat/completions",
            status_code=result["status_code"],
            request=request_body,
            response=response,
        )
        appended = manager.append_session_record(session_id, record)
        if appended is None:
            return JSONResponse(status_code=404, content={"error": "session not found"})
        return router._build_proxy_response(result)

    @app.api_route("/sessions/{session_id}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def session_proxy(request: Request, session_id: str, path: str):
        result = await router._do_proxy(request, path)
        return router._build_proxy_response(result)
