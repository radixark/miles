"""Train from actual renderer/SDK samples, including an edited history and a repeated prompt."""

import argparse
import asyncio

import httpx
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.rl.data_processing import trajectory_to_data

from miles.tinker.client.rendering import ChatRequest, render_prompt
from miles.tinker.client.server import SessionServer
from miles.tinker.client.session import ChatSession
from miles.tinker.client.trajectory import turns_to_trajectory


async def execute(base_url, base_model):
    service = tinker.ServiceClient(base_url=base_url, api_key="tml-session-acceptance")
    training = await service.create_lora_training_client_async(base_model=base_model, rank=8)
    saved = await (await training.save_weights_for_sampler_async(name="session-start"))
    sampler = await service.create_sampling_client_async(model_path=saved.path)
    renderer = renderers.get_renderer("qwen3_instruct", sampler.get_tokenizer())
    server = SessionServer()
    datums = []
    async with server.serve() as port, httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=300) as http:
        for advantage in [-1.0, 1.0]:
            session = ChatSession(TinkerTokenCompleter(sampler, max_tokens=32), renderer, 2048)
            async with server.session(session) as path:
                messages = [{"role": "user", "content": "Reply with the number 1."}]
                for index in range(3):
                    request = {"messages": messages, "max_tokens": 32}
                    response = await http.post(f"{path}/v1/chat/completions", json=request)
                    response.raise_for_status()
                    assert list(session.trace.turns[-1].input_ids) == render_prompt(renderer, ChatRequest(**request))
                    if index == 0:
                        messages = [
                            {"role": "user", "content": "Reply with the number 2."},
                            response.json()["choices"][0]["message"],
                            {"role": "user", "content": "Reply with the number 3."},
                        ]
            trajectory = turns_to_trajectory(session.trace.turns)
            group_datums = trajectory_to_data(trajectory, traj_advantage=advantage)
            assert len(group_datums) == 3
            assert sum(sum(d.loss_fn_inputs["mask"].data) for d in group_datums) == sum(
                len(turn.output_ids) for turn in session.trace.turns
            )
            datums.extend(group_datums)
    backward = await training.forward_backward_async(datums, loss_fn="ppo")
    step = await training.optim_step_async(tinker.types.AdamParams(learning_rate=1e-5))
    await backward
    await step
    await (await training.save_weights_for_sampler_async(name="session-trained"))
    print("session acceptance passed: 6 sampled turns, 6 Datums, PPO backward + optimizer step")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--base-model", required=True)
    args = parser.parse_args()
    asyncio.run(execute(args.base_url, args.base_model))
