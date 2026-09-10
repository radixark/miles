"""Serve NeMo Gym's stateless Sokoban verifier without starting a second Ray cluster."""

from fastapi import FastAPI
from omegaconf import OmegaConf

from nemo_gym.server_utils import ServerClient
from resources_servers.reasoning_gym.app import ReasoningGymResourcesServer, ReasoningGymResourcesServerConfig


def create_app() -> FastAPI:
    client = ServerClient(
        head_server_config={"host": "127.0.0.1", "port": 0},
        global_config_dict=OmegaConf.create({}),
    )
    server = ReasoningGymResourcesServer(
        config=ReasoningGymResourcesServerConfig(
            name="sokoban",
            host="127.0.0.1",
            port=8210,
            entrypoint="app.py",
        ),
        server_client=client,
    )
    app = server.setup_webserver()
    server.setup_liveness(app)
    return app
