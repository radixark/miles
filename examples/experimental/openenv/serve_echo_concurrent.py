"""Self-hosted OpenEnv Echo server with concurrent sessions for the miles smoke run.

The packaged Echo server (``echo_env.server.app``) caps at a single session
(``SUPPORTS_CONCURRENT_SESSIONS=False`` -> ``max_concurrent_envs=1``), which
collides with miles' concurrent rollouts (rollout_batch_size * n_samples). Echo's
reward is constant and ``create_app`` instantiates a *fresh* environment per
WebSocket session (the class is passed as a factory), so each session is isolated
and enabling concurrency is safe.

Run colocated with the training job and point the launcher at it:

    python serve_echo_concurrent.py --port 8001 --max-concurrent 64
    python run-openenv-echo.py --openenv-env-url http://localhost:8001 ...
"""

import uvicorn
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from tap import Tap

from echo_env.server.echo_environment import EchoEnvironment


class ConcurrentEchoEnvironment(EchoEnvironment):
    SUPPORTS_CONCURRENT_SESSIONS = True


class Args(Tap):
    host: str = "0.0.0.0"
    port: int = 8001
    max_concurrent: int = 64


def main() -> None:
    args = Args().parse_args()
    app = create_app(
        ConcurrentEchoEnvironment,
        CallToolAction,
        CallToolObservation,
        env_name="echo_env",
        max_concurrent_envs=args.max_concurrent,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
