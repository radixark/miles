"""Train from the official proxy's captured tokens, including edited and repeated histories."""

import argparse
import asyncio
import tempfile

import httpx
from tinker_agent.proxy import serve_proxy
from tinker_agent.trajectory import flush_capture, read_samples, rows_to_trajectory
from tinker_cookbook import renderers
from tinker_cookbook.capture.store import capture_to_store
from tinker_cookbook.rl.data_processing import trajectory_to_data

import tinker


async def execute(base_url, base_model, store_url):
    service = tinker.ServiceClient(base_url=base_url, api_key="tml-session-acceptance")
    training = await service.create_lora_training_client_async(base_model=base_model, rank=8)
    saved = await (await training.save_weights_for_sampler_async(name="session-start"))
    sampler = await service.create_sampling_client_async(model_path=saved.path)
    renderer = renderers.get_renderer("qwen3_instruct", sampler.get_tokenizer())
    async with serve_proxy(renderer, sampler, model_label=base_model, default_max_tokens=32) as port:
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=300) as http:
            for trajectory_id in range(2):
                messages = [{"role": "user", "content": "Reply with the number 1."}]
                for index in range(3):
                    response = await http.post(
                        f"/r/run/acceptance/traj/{trajectory_id}/v1/chat/completions",
                        json={"messages": messages, "max_tokens": 32},
                    )
                    response.raise_for_status()
                    if index == 0:
                        messages = [
                            {"role": "user", "content": "Reply with the number 2."},
                            response.json()["choices"][0]["message"],
                            {"role": "user", "content": "Reply with the number 3."},
                        ]
    await flush_capture()
    datums = []
    for trajectory_id, advantage in enumerate([-1.0, 1.0]):
        rows = await read_samples(store_url, "acceptance", traj_idx=trajectory_id)
        assert len(rows) == 3
        group_datums = trajectory_to_data(rows_to_trajectory(rows, "completed"), traj_advantage=advantage)
        assert len(group_datums) == 3
        assert sum(sum(d.loss_fn_inputs["mask"].data) for d in group_datums) == sum(
            len(row["sampled_tokens"]) for row in rows
        )
        datums.extend(group_datums)
    # The cookbook trainer drops its client-side "mask" before forward_backward (rl/train.py _remove_mask).
    sent = [
        tinker.Datum(
            model_input=datum.model_input,
            loss_fn_inputs={key: value for key, value in datum.loss_fn_inputs.items() if key != "mask"},
        )
        for datum in datums
    ]
    backward = await training.forward_backward_async(sent, loss_fn="ppo")
    step = await training.optim_step_async(tinker.types.AdamParams(learning_rate=1e-5))
    await backward
    await step
    trained = await (await training.save_weights_for_sampler_async(name="session-trained"))
    updated_sampler = await service.create_sampling_client_async(model_path=trained.path)
    response = await updated_sampler.sample_async(
        prompt=sent[0].model_input,
        num_samples=1,
        sampling_params=tinker.SamplingParams(max_tokens=4),
    )
    assert response.sequences[0].tokens
    assert len(response.sequences[0].tokens) == len(response.sequences[0].logprobs)
    print("capture acceptance passed: 6 samples from official store, 6 Datums, PPO step, updated-policy sampling")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--base-model", required=True)
    args = parser.parse_args()
    with capture_to_store(
        "acceptance",
        data_dir=tempfile.mkdtemp(prefix="tinker-capture-"),
        idle_shutdown_minutes=1,
    ) as captured:
        asyncio.run(execute(args.base_url, args.base_model, captured.base_url))
