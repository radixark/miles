"""Self-hosted OpenEnv Coding server with concurrent sessions for the miles run.

The packaged Coding server (``coding_env.server.app``) caps at a single session
(``SUPPORTS_CONCURRENT_SESSIONS=False`` -> ``max_concurrent_envs=1``), which
collides with miles' concurrent rollouts (rollout_batch_size * n_samples).
``create_app`` instantiates a *fresh* ``PythonCodeActEnv`` per WebSocket session
(the class is passed as a factory) and each env owns its own in-process
``PyExecutor``, so sessions are isolated and enabling concurrency is safe.

Unlike Echo, the Coding env's reward is a real signal: the safe-coding transform
penalizes dangerous patterns (-1.0) and syntax errors (-0.2) and rewards concise
code (+0.1), so raw_reward should climb as the policy learns to emit short, safe,
valid Python -- this exercises GRPO *learning* through the seam (Phase 2).

Run colocated with the training job and point the launcher at it:

    python serve_coding_concurrent.py --port 8002 --max-concurrent 64
    python run-openenv-coding.py --openenv-env-url http://localhost:8002 ...
"""

import uvicorn
from openenv.core.env_server import create_app
from tap import Tap

from coding_env.models import CodeAction, CodeObservation
from coding_env.server.python_codeact_env import PythonCodeActEnv


class ConcurrentPythonCodeActEnv(PythonCodeActEnv):
    SUPPORTS_CONCURRENT_SESSIONS = True


class Args(Tap):
    host: str = "0.0.0.0"
    port: int = 8002
    max_concurrent: int = 64


def main() -> None:
    args = Args().parse_args()
    app = create_app(
        ConcurrentPythonCodeActEnv,
        CodeAction,
        CodeObservation,
        env_name="coding_env",
        max_concurrent_envs=args.max_concurrent,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
