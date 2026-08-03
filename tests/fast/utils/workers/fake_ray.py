from __future__ import annotations

import asyncio
import itertools
import time
from dataclasses import dataclass, field, replace
from typing import Any

_ASYNC_METHOD_NODE_IP = "_get_node_ip"
_ASYNC_METHOD_FREE_PORT_BLOCK = "_get_free_port_block"

EVENT_CREATE = "create"
EVENT_KILL = "kill"


@dataclass(kw_only=True)
class FakeRayObjectRef:
    method: str
    value: Any = None
    error: BaseException | None = None
    hang_seconds: float | None = None

    def __await__(self):
        return self._resolve_async().__await__()

    async def _resolve_async(self) -> Any:
        if self.hang_seconds is not None:
            await asyncio.sleep(self.hang_seconds)
        return self.resolve()

    def resolve(self) -> Any:
        if self.error is not None:
            raise self.error
        return self.value


@dataclass(kw_only=True)
class FakeRayActorCall:
    handle: FakeRayActorHandle
    method: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


@dataclass(kw_only=True)
class FakeRayActorMethod:
    handle: FakeRayActorHandle
    method: str

    def remote(self, *args: Any, **kwargs: Any) -> FakeRayObjectRef:
        return self.handle.cluster.dispatch(handle=self.handle, method=self.method, args=args, kwargs=kwargs)


@dataclass(kw_only=True)
class FakeRayActorHandle:
    cluster: FakeRayCluster
    actor_class: type
    index: int
    options: dict[str, Any]
    node_ip: str
    failing_methods: dict[str, BaseException] = field(default_factory=dict)
    hanging_methods: dict[str, float] = field(default_factory=dict)
    killed: bool = False

    def __getattr__(self, name: str) -> FakeRayActorMethod:
        if name.startswith("__") or "cluster" not in self.__dict__:
            raise AttributeError(name)
        return FakeRayActorMethod(handle=self, method=name)


@dataclass(kw_only=True)
class FakeRayRemoteClass:
    cluster: FakeRayCluster
    actor_class: type
    actor_options: dict[str, Any] = field(default_factory=dict)

    def options(self, **kwargs: Any) -> FakeRayRemoteClass:
        return replace(self, actor_options=kwargs)

    def remote(self, *args: Any, **kwargs: Any) -> FakeRayActorHandle:
        assert not args, "actor constructors must be called with keyword arguments"
        handle = self.cluster.create_actor(
            actor_class=self.actor_class, options=self.actor_options, ctor_kwargs=kwargs
        )
        if (name := self.actor_options.get("name")) is not None:
            self.cluster.named_actors[name] = handle
        return handle


@dataclass(kw_only=True)
class FakeRayModule:
    cluster: FakeRayCluster

    def remote(self, actor_class: type) -> FakeRayRemoteClass:
        return FakeRayRemoteClass(cluster=self.cluster, actor_class=actor_class)

    def get(self, ref: FakeRayObjectRef, timeout: float | None = None) -> Any:
        self.cluster.resolved_refs.append(ref.method)
        self.cluster.get_timeouts.append(timeout)
        if ref.hang_seconds is not None:
            time.sleep(ref.hang_seconds)
        return ref.resolve()

    def kill(self, handle: FakeRayActorHandle, no_restart: bool = False) -> None:
        self.cluster.kill_actor(handle)

    def get_actor(self, name: str) -> FakeRayActorHandle:
        return self.cluster.named_actors[name]


@dataclass
class FakeRayCluster:
    node_ips: tuple[str, ...] = ("10.0.0.1",)
    base_port: int = 15000
    handles: list[FakeRayActorHandle] = field(default_factory=list)
    calls: list[FakeRayActorCall] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    named_actors: dict[str, FakeRayActorHandle] = field(default_factory=dict)
    kill_error: BaseException | None = None
    method_errors: dict[str, BaseException] = field(default_factory=dict)
    resolved_refs: list[str] = field(default_factory=list)
    get_timeouts: list[float | None] = field(default_factory=list)
    ctor_kwargs: list[dict[str, Any]] = field(default_factory=list)
    _used_ports: dict[str, set[int]] = field(default_factory=dict)
    _node_ip_cycle: Any = None

    def __post_init__(self) -> None:
        self._node_ip_cycle = itertools.cycle(self.node_ips)

    def occupy_ports(self, node_ip: str, *ports: int) -> None:
        """Stand in for a process from an earlier run still listening on the node."""
        self._used_ports.setdefault(node_ip, set()).update(ports)

    def use_node_ips(self, *node_ips: str) -> None:
        self.node_ips = node_ips
        self._node_ip_cycle = itertools.cycle(node_ips)

    def create_actor(
        self, *, actor_class: type, options: dict[str, Any], ctor_kwargs: dict[str, Any]
    ) -> FakeRayActorHandle:
        handle = FakeRayActorHandle(
            cluster=self,
            actor_class=actor_class,
            index=len(self.handles),
            options=options,
            node_ip=next(self._node_ip_cycle),
        )
        self.handles.append(handle)
        self.ctor_kwargs.append(ctor_kwargs)
        self.events.append(EVENT_CREATE)
        return handle

    def kill_actor(self, handle: FakeRayActorHandle) -> None:
        self.events.append(EVENT_KILL)
        if self.kill_error is not None:
            raise self.kill_error
        handle.killed = True

    def dispatch(
        self, *, handle: FakeRayActorHandle, method: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> FakeRayObjectRef:
        self.calls.append(FakeRayActorCall(handle=handle, method=method, args=args, kwargs=kwargs))
        self.events.append(method)
        return FakeRayObjectRef(
            method=method,
            value=self._compute_value(handle=handle, method=method, kwargs=kwargs),
            error=handle.failing_methods.get(method, self.method_errors.get(method)),
            hang_seconds=handle.hanging_methods.get(method),
        )

    def calls_of(self, method: str) -> list[FakeRayActorCall]:
        return [call for call in self.calls if call.method == method]

    def first_event_index(self, event: str) -> int:
        return self.events.index(event)

    def last_event_index(self, event: str) -> int:
        return len(self.events) - 1 - self.events[::-1].index(event)

    def _compute_value(self, *, handle: FakeRayActorHandle, method: str, kwargs: dict[str, Any]) -> Any:
        if method == _ASYNC_METHOD_NODE_IP:
            return handle.node_ip
        if method == _ASYNC_METHOD_FREE_PORT_BLOCK:
            return self._alloc_port_block(
                node_ip=handle.node_ip, start_port=kwargs["start_port"], count=kwargs["count"]
            )
        return None

    def _alloc_port_block(self, *, node_ip: str, start_port: int, count: int) -> int:
        used = self._used_ports.setdefault(node_ip, set())
        # Exactly start_port when it is free, like the real get_free_port: the floor lives in
        # PortAllocator, and clamping here would make a pinned port look occupied.
        port = start_port
        while any(port + offset in used for offset in range(count)):
            port += 1
        used.update(range(port, port + count))
        return port
