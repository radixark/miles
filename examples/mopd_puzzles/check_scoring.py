"""Compare sparse teacher scores against SGLang's dense requested-ID oracle."""

import argparse
import json
import time
from pathlib import Path

import httpx
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    row = json.loads(args.data.read_text().splitlines()[0])
    prompt = tokenizer.apply_chat_template(
        row["prompt"], tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False
    )
    with httpx.Client(timeout=600) as client:
        response = client.post(
            args.url + "/generate",
            json=dict(
                input_ids=prompt,
                sampling_params=dict(temperature=1, max_new_tokens=128),
                return_logprob=True,
                top_logprobs_num=16,
            ),
        )
        response.raise_for_status()
        meta = response.json()["meta_info"]
        response_ids = [entry[1] for entry in meta["output_token_logprobs"]]
        candidate_rows = [[entry[1] for entry in entries] for entries in meta["output_top_logprobs"]]
        report = []
        for length in [len(response_ids), 512]:
            tokens = prompt + [response_ids[i % len(response_ids)] for i in range(length)]
            rows = [candidate_rows[i % len(candidate_rows)] for i in range(length)]
            ids = sorted({token for candidates in rows for token in candidates})
            base = dict(
                input_ids=tokens,
                sampling_params=dict(temperature=0, max_new_tokens=0),
                return_logprob=True,
                logprob_start_len=0,
            )
            results = {}
            for mode, fields in [
                ("dense", dict(token_ids_logprob=ids)),
                ("sparse", dict(token_ids_logprob_positions=[[] for _ in prompt] + rows)),
            ]:
                start = time.monotonic()
                value = client.post(args.url + "/generate", json={**base, **fields})
                value.raise_for_status()
                elapsed = time.monotonic() - start
                meta = value.json()["meta_info"]
                scored = meta["input_token_ids_logprobs"]
                if len(scored) != len(tokens):
                    raise AssertionError(f"{mode}: {len(scored)} score positions for {len(tokens)} input tokens")
                results[mode] = dict(
                    seconds=elapsed,
                    bytes=len(value.content),
                    scores=scored[-length:],
                    prompt_scores=scored[1 : len(prompt)],
                )
            max_error = 0.0
            for candidates, dense, sparse in zip(
                rows, results["dense"]["scores"], results["sparse"]["scores"], strict=True
            ):
                if [entry[1] for entry in sparse] != candidates:
                    raise AssertionError("Sparse candidate positions/order drifted")
                oracle = {entry[1]: entry[0] for entry in dense}
                max_error = max(max_error, *(abs(entry[0] - oracle[entry[1]]) for entry in sparse))
            if max_error > 2e-3:
                raise AssertionError(f"Dense/sparse scoring disagreement: {max_error}")
            if any(results["sparse"]["prompt_scores"]):
                raise AssertionError("Sparse request returned unwanted prompt candidates")
            report.append(
                dict(
                    response_length=length,
                    prompt_length=len(prompt),
                    unique_ids=len(ids),
                    max_abs_error=max_error,
                    **{mode: {k: results[mode][k] for k in ["seconds", "bytes"]} for mode in ["dense", "sparse"]},
                )
            )
            print(json.dumps(report[-1]), flush=True)
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
