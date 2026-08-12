"""One real episode against a real sandbox, with a scripted policy.

The offline tests cover the translation logic; this covers everything they
cannot: the image booting, MCP coming up, the task row's setup and grade calls
working, and screenshots arriving as usable images. Run it before launching a
training job, and after changing the image or the taskset.

    python -m examples.experimental.hud.smoke_episode --dataset hud-evals/2048-basic
"""

import argparse
import json
import time
from types import SimpleNamespace

from examples.experimental.hud.hud_task_env import HudTaskEnv
from examples.experimental.hud.make_hud_data import fetch_rows, task_of

# Arrow keys, because they are the one vocabulary safe to send at any GUI
# without changing state in a way that confuses the row's own grader.
SCRIPT = [
    "Let me look at the screen.\nAction: screenshot()",
    'Move around.\nAction: press("Left","Down","Left","Down")',
    'Again.\nAction: press("Left","Down","Left","Down")',
    "this turn has no action line at all",  # parse-failure path
    'Single key, should be flagged as short.\nAction: press("Down")',
    'Too many keys, should be clipped.\nAction: press("Left","Down","Left","Down","Left","Down")',
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hud-evals/2048-basic")
    # Off by default, so this works on any taskset: with no surrogate the row's
    # own evaluate_tool is the reward, which every row has.
    ap.add_argument("--target-score", type=int, help="2048 only: also grade with game_2048_score_reached")
    args = ap.parse_args()

    row = fetch_rows(args.dataset, "train", 1)[0]
    task = task_of(row)
    print(f"task from {args.dataset}: setup={json.dumps(task['setup_tool'])[:80]}")

    env = HudTaskEnv(
        SimpleNamespace(metadata={"hud_task": task}),
        SimpleNamespace(
            hud_screenshot_width=640,
            hud_keys_per_turn=4,
            hud_max_sandboxes=4,
            hud_sandbox_max_age_min=20,
            hud_reward_tool=(
                {
                    "name": "evaluate",
                    "arguments": {
                        "name": "game_2048_score_reached",
                        "arguments": {"target_score": args.target_score},
                    },
                }
                if args.target_score
                else None
            ),
        ),
    )

    t0 = time.time()
    try:
        _, info = env.reset()
        print(f"[{time.time() - t0:5.1f}s] reset: {info}")
        for i, response in enumerate(SCRIPT, 1):
            obs, done, step_info = env.step(response)
            images = (obs.get("multi_modal_data") or {}).get("image", [])
            print(
                f"[{time.time() - t0:5.1f}s] step {i}: done={done} img={images[0].size if images else None} "
                f"actions={env.actions} obs={obs.get('obs_str', '')[:58]}"
            )
            if done:
                break
        verdict = env.compute_final_reward()
        print(f"[{time.time() - t0:5.1f}s] verdict: {json.dumps(verdict, default=str)}")

        assert 0.0 <= verdict["reward"] <= 1.0, verdict
        assert "task_reward" in verdict, "the taskset's own grade must always be recorded"
        assert verdict["actions"] >= 9, f"scripted policy should make >=9 moves, got {verdict['actions']}"
        assert verdict["short_actions"] == 1, verdict
        print("SMOKE EPISODE PASSED")
    finally:
        env.close()
        print(f"[{time.time() - t0:5.1f}s] sandbox released")


if __name__ == "__main__":
    main()
