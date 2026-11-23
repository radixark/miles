import asyncio

import aiohttp
from tinker_cookbook.capture.instrument import current_exporter
from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.types import Trajectory, Transition

import tinker


async def flush_capture():
    exporter = current_exporter()
    assert exporter is not None, "run the recipe inside capture_to_store"
    flushed = await asyncio.to_thread(exporter.force_flush)
    if not flushed or exporter.dropped or exporter.export_failures or exporter.callback_failures:
        raise RuntimeError(
            f"capture data is incomplete: flushed={flushed}, dropped={exporter.dropped}, "
            f"export_failures={exporter.export_failures}, callback_failures={exporter.callback_failures}"
        )


async def read_samples(store_url, run_id, **coordinates):
    rows = []
    cursor = 0
    async with aiohttp.ClientSession() as http:
        while True:
            async with http.get(
                f"{store_url}/runs/{run_id}/rows", params={**coordinates, "cursor": cursor, "limit": 1000}
            ) as response:
                response.raise_for_status()
                page = await response.json()
            rows.extend(page["rows"])
            cursor = page["next_cursor"]
            if cursor is None:
                break
    # Store cursors follow completion/ingest order; capture timestamps record call time.
    return sorted(rows, key=lambda row: (row["created_at"], row["cursor"]))


def rows_to_trajectory(rows, stop_reason: str) -> Trajectory:
    transitions = []
    for index, row in enumerate(rows):
        assert row["metadata"]["status"] == "ok"
        assert row["metadata"]["num_samples"] == 1
        assert row["logprobs"] is not None and len(row["sampled_tokens"]) == len(row["logprobs"])
        transitions.append(
            Transition(
                ob=tinker.ModelInput.from_ints(row["prompt_tokens"]),
                ac=TokensWithLogprobs(
                    tokens=row["sampled_tokens"],
                    maybe_logprobs=row["logprobs"],
                    stop_reason=row["metadata"]["stop_reason"],
                ),
                reward=0.0,
                episode_done=index == len(rows) - 1,
            )
        )
    if not transitions:
        transitions = [
            Transition(
                ob=tinker.ModelInput.empty(),
                ac=TokensWithLogprobs(tokens=[], maybe_logprobs=[]),
                reward=0.0,
                episode_done=True,
            )
        ]
    final_ob = (
        tinker.ModelInput.from_ints([*rows[-1]["prompt_tokens"], *rows[-1]["sampled_tokens"]])
        if rows
        else tinker.ModelInput.empty()
    )
    return Trajectory(transitions=transitions, final_ob=final_ob, stop_reason=stop_reason)
