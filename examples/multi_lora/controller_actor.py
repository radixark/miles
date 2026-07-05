"""Ray actor wrapper around MultiLoRAControllerLogic + MultiLoRAHTTPServer.

Kept separate from controller.py so the logic + HTTP server stay Ray-free and
unit-testable. The actor runs an HTTP server out-of-band (Ray docs pattern:
``asyncio.get_running_loop().create_task``) so:

  - trainer / datasource reach it via Ray (register/deregister/active),
  - rollout requests reach it via HTTP (the proxy at the port returned by start).

The Ray methods and the HTTP handlers share the same in-process
``MultiLoRAControllerLogic`` instance (single source of truth).
"""

import ray

from typing import Any

from examples.multi_lora.controller import MultiLoRAControllerLogic, MultiLoRAHTTPServer


@ray.remote
class MultiLoRAController:
    def __init__(self, upstream_url: str, host: str = "0.0.0.0", port: int = 0) -> None:
        self.logic = MultiLoRAControllerLogic()
        self.server = MultiLoRAHTTPServer(self.logic, upstream_url, host, port)

    async def start(self) -> int:
        await self.server.start()
        return self.server.actual_port

    async def stop(self) -> None:
        await self.server.stop()

    def register(self, name: str, slot: int, config: Any = None) -> None:
        self.logic.register(name, slot, config)

    def deregister(self, name: str) -> None:
        self.logic.deregister(name)

    def active(self) -> dict:
        return self.logic.active()

    def active_adapters(self) -> dict:
        return self.logic.active_adapters()

    def http_host(self) -> str:
        return self.server.host

    def http_port(self) -> int:
        return self.server.actual_port
