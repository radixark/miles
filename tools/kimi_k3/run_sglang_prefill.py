import argparse
import json
from pathlib import Path

import sglang as sgl

from miles.utils.debug_utils.run_megatron.sglang_output import build_sglang_logprob_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save SGLang prompt logprobs in run_megatron format.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--token-ids-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--mem-fraction-static", type=float, default=0.85)
    parser.add_argument("--moe-runner-backend", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token_ids = json.loads(args.token_ids_file.read_text())
    assert isinstance(token_ids, list) and len(token_ids) >= 2
    assert all(type(token_id) is int for token_id in token_ids)

    engine = sgl.Engine(
        model_path=str(args.model_path),
        tp_size=args.tp_size,
        trust_remote_code=True,
        skip_tokenizer_init=True,
        mem_fraction_static=args.mem_fraction_static,
        moe_runner_backend=args.moe_runner_backend,
        max_running_requests=1,
        enable_deterministic_inference=True,
        disable_cuda_graph=True,
    )
    try:
        result = engine.generate(
            input_ids=token_ids,
            sampling_params={"temperature": 0.0, "max_new_tokens": 1},
            return_logprob=True,
            logprob_start_len=0,
        )
    finally:
        engine.shutdown()

    assert result is not None, "SGLang returned no generation result"
    if isinstance(result, list):
        assert len(result) == 1
        result = result[0]
    payload = build_sglang_logprob_payload(
        token_ids,
        result["meta_info"]["input_token_logprobs"],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "rank_0.json"
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved {len(token_ids) - 1} SGLang prompt logprobs to {output_path}", flush=True)


if __name__ == "__main__":
    main()
