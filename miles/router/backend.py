"""Router implementations behind one interface: the launch spec owns the router process, a backend owns its command."""

import enum
import logging
from abc import ABC, abstractmethod
from argparse import Namespace
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from miles.backends.sglang_utils.router_args_utils import (
    compute_sglang_router_args,
    resolve_router_policies,
    router_args_to_argv,
)
from miles.router.config import compute_miles_router_config
from miles.utils.http_utils import resolve_ip
from miles.utils.workers.argv_utils import config_to_argv
from miles.utils.workers.worker_spec import LaunchCommandContext

if TYPE_CHECKING:
    # sglang_config imports the sglang arguments module, which reads the router backend
    from miles.backends.sglang_utils.sglang_config import ModelConfig

logger = logging.getLogger(__name__)


class RouterKind(str, enum.Enum):
    SGLANG = "sglang"
    MILES = "miles"


@dataclass(frozen=True)
class RouterFeatures:
    pd_disaggregation: bool
    # header that carries a request's routing key; None = the router never routes by key
    routing_key_header: str | None


class RouterBackend(ABC):
    kind: ClassVar[RouterKind]
    features: ClassVar[RouterFeatures]

    def __init__(self, args: Namespace) -> None:
        self.args = args

    def launch_argv(
        self, ctx: LaunchCommandContext, *, model_cfg: "ModelConfig", interpreter_prefix: list[str]
    ) -> list[str]:
        needs_pd_disaggregation = self._needs_pd_disaggregation(model_cfg)
        assert (
            self.features.pd_disaggregation or not needs_pd_disaggregation
        ), f"{self.kind.value} router does not support PD disaggregation."
        return self._launch_argv(ctx, model_cfg=model_cfg, interpreter_prefix=interpreter_prefix)

    def _needs_pd_disaggregation(self, model_cfg: "ModelConfig | None") -> bool:
        # None: an external router in front of engines Miles did not launch
        return self.args.rollout_external_router_pd or (model_cfg is not None and model_cfg.has_pd_disaggregation)

    @abstractmethod
    def _launch_argv(
        self, ctx: LaunchCommandContext, *, model_cfg: "ModelConfig", interpreter_prefix: list[str]
    ) -> list[str]: ...

    def routes_by_key(self, model_cfgs: "Sequence[ModelConfig]") -> bool:
        """Whether any router in front of `model_cfgs` places requests by routing key.

        Each model has its own router and requests may target any of them, so a key goes out whenever one
        router reads it; the others ignore the header. No models means an external router.
        """
        return any(self._router_routes_by_key(model_cfg) for model_cfg in model_cfgs or [None])

    @abstractmethod
    def _router_routes_by_key(self, model_cfg: "ModelConfig | None") -> bool: ...

    def routing_headers(self, routing_key: str | None, model_cfgs: "Sequence[ModelConfig]") -> dict[str, str]:
        if not self.routes_by_key(model_cfgs):
            return {}
        if not routing_key:
            raise ValueError(f"the {self.kind.value} router routes by routing key, but the request carries none")
        return {self.features.routing_key_header: routing_key}


class SGLangRouterBackend(RouterBackend):
    kind = RouterKind.SGLANG
    features = RouterFeatures(pd_disaggregation=True, routing_key_header="X-SMG-Routing-Key")
    # sgl-router answers 400 missing_routing_key under these without the header
    _KEY_POLICIES = frozenset({"consistent_hashing", "manual"})

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self._policies = resolve_router_policies(args)

    def _launch_argv(
        self, ctx: LaunchCommandContext, *, model_cfg: "ModelConfig", interpreter_prefix: list[str]
    ) -> list[str]:
        primary = ctx.self_addrs["primary"]
        router_args = compute_sglang_router_args(
            self.args,
            host=resolve_ip(primary.host),
            port=primary.port,
            prometheus_port=ctx.self_addrs["prometheus"].port,
            has_pd_disaggregation=self._needs_pd_disaggregation(model_cfg),
        )
        logger.info(f"Launch router with args: {router_args}")
        return [*interpreter_prefix, "-m", "sglang_router.launch_router", *router_args_to_argv(router_args)]

    def _router_routes_by_key(self, model_cfg: "ModelConfig | None") -> bool:
        main = self._policies["policy"]
        if self._needs_pd_disaggregation(model_cfg):
            # each PD pool runs its own policy, falling back to the main one
            in_effect = {self._policies["prefill_policy"] or main, self._policies["decode_policy"] or main}
        else:
            in_effect = {main}
        return not self._KEY_POLICIES.isdisjoint(in_effect)


class MilesRouterBackend(RouterBackend):
    kind = RouterKind.MILES
    features = RouterFeatures(pd_disaggregation=False, routing_key_header=None)

    def _launch_argv(
        self, ctx: LaunchCommandContext, *, model_cfg: "ModelConfig", interpreter_prefix: list[str]
    ) -> list[str]:
        primary = ctx.self_addrs["primary"]
        router_config = compute_miles_router_config(self.args, host=primary.host, port=primary.port)
        return [*interpreter_prefix, "-m", "miles.router.router", *config_to_argv(router_config)]

    def _router_routes_by_key(self, model_cfg: "ModelConfig | None") -> bool:
        return False


_BACKEND_CLASSES: dict[RouterKind, type[RouterBackend]] = {
    RouterKind.SGLANG: SGLangRouterBackend,
    RouterKind.MILES: MilesRouterBackend,
}


def make_router_backend(args: Namespace) -> RouterBackend:
    return _BACKEND_CLASSES[RouterKind(args.router)](args)
