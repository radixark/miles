"""Ray actor wrapper around MultiLoRAControllerLogic + MultiLoRAHTTPServer.

Kept separate from controller.py so the logic + HTTP server stay Ray-free and
unit-testable. The actor runs an HTTP server out-of-band (Ray docs pattern:
``asyncio.get_running_loop().create_task``) so:

  - trainer / datasource reach it via Ray (register/retire/active),
  - rollout requests reach it via HTTP (the proxy at the port returned by start).

The Ray methods and the HTTP handlers share the same in-process
``MultiLoRAControllerLogic`` instance (single source of truth).
"""

import ray

from examples.multi_lora.controller import MultiLoRAControllerLogic, MultiLoRAHTTPServer


@ray.remote
class MultiLoRAController:
    def __init__(self, upstream_url: str, host: str = "0.0.0.0", port: int = 0) -> None:
        self.logic = MultiLoRAControllerLogic()
        self._srv = MultiLoRAHTTPServer(self.logic, upstream_url, host, port)

    async def start(self) -> int:
        await self._srv.start()
        return self._srv.actual_port

    async def stop(self) -> None:
        await self._srv.stop()

    def register(self, name: str, slot: int) -> None:
        self.logic.register(name, slot)

    def retire(self, name: str) -> None:
        self.logic.retire(name)

    def active(self) -> dict:
        return self.logic.active()

    def http_host(self) -> str:
        return self._srv.host

    def http_port(self) -> int:
        return self._srv.actual_port
