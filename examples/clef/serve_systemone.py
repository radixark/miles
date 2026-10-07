"""Local SystemOne gateway to SGLang's trained Clef prefill adapter."""

import asyncio
import hashlib
import json
import math
import os
import uuid
from dataclasses import asdict
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from tap import Tap
from transformers import AutoTokenizer

from examples.clef.joint_schema_model import encode_record, systemone_answer


class Args(Tap):
    model_path: Path
    metadata_dir: Path
    engine_url: str = "http://127.0.0.1:31000"
    host: str = "0.0.0.0"
    port: int = 31001
    max_length: int = 16384


def create_app(args: Args) -> FastAPI:
    app = FastAPI()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    args.metadata_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(1)

    @app.get("/health")
    async def health() -> dict:
        # SGLang's embedding health probe carries no schema spans. Send an
        # actual schema request so readiness also checks the trained head.
        await systemone({"model": "clef", "state": "Readiness check.",
                         "questions": {"ready": {"type": "noul", "instructions": "Is this a readiness check?"}}})
        return {"status": "ok", "model_path": str(args.model_path)}

    @app.post("/v1/systemone")
    async def systemone(request: dict) -> dict:
        if "state" not in request or not isinstance(request.get("questions"), dict) or not request["questions"]:
            raise HTTPException(400, "state and a nonempty questions mapping are required")
        if request.get("images") or request.get("videos"):
            raise HTTPException(400, "This Clef gateway supports text requests only")
        try:
            encoded = encode_record(tokenizer, request, max_length=args.max_length)
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        data = asdict(encoded)
        key = hashlib.sha256(json.dumps(data["input_ids"], separators=(",", ":")).encode()).hexdigest()
        target = args.metadata_dir / (key + ".json")
        temporary = target.with_suffix("." + uuid.uuid4().hex + ".tmp")
        temporary.write_text(json.dumps(data))
        os.replace(temporary, target)
        async with semaphore, httpx.AsyncClient(timeout=600) as client:
            result = await client.post(args.engine_url + "/encode", json={"input_ids": data["input_ids"]})
            result.raise_for_status()
        output = result.json()
        if isinstance(output, list):
            output = output[0]
        values = output["embedding"]
        expected_count = sum(len(q.option_ids) for q in encoded.questions)
        if len(values) != expected_count:
            raise HTTPException(502, "SGLang returned an incorrect option count")
        answers, full_probabilities = {}, {}
        offset = 0
        for question in encoded.questions:
            count = len(question.option_ids)
            probabilities = dict(zip(question.option_ids, values[offset : offset + count], strict=True))
            if any(not math.isfinite(p) or p < 0 for p in probabilities.values()) or abs(sum(probabilities.values()) - 1) > 1e-5:
                raise HTTPException(502, "SGLang returned an invalid probability distribution")
            answers[question.question_id] = systemone_answer(request["questions"][question.question_id], probabilities)
            full_probabilities[question.question_id] = probabilities
            offset += count
        return {"model": request.get("model", "clef"), "answers": answers,
                "probabilities": full_probabilities,
                "usage": {"input_tokens": len(encoded.input_ids), "output_tokens": 0}}

    return app


if __name__ == "__main__":
    arguments = Args(underscores_to_dashes=True).parse_args()
    uvicorn.run(create_app(arguments), host=arguments.host, port=arguments.port)
