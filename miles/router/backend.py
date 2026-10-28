"""Router implementations behind one interface: the launch spec owns the router process, a backend owns its command."""

import enum
import logging
from abc import ABC, abstractmethod
from argparse import Namespace
from dataclasses import dataclass
from typing import ClassVar

from miles.backends.sglang_utils.router_args_utils import compute_sglang_router_args, router_args_to_argv
from miles.backends.sglang_utils.sglang_config import ModelConfig
from miles.router.config import compute_miles_router_config
from miles.utils.http_utils import resolve_ip
from miles.utils.workers.argv_utils import config_to_argv
from miles.utils.workers.worker_spec import LaunchCommandContext

logger = logging.getLogger(__name__)


class RouterKind(str, enum.Enum):
    SGLANG = "sglang"
    MILES = "miles"


@dataclass(frozen=True)
class RouterFeatures:
    pd_disaggregation: bool


class RouterBackend(ABC):
    kind: ClassVar[RouterKind]
    features: ClassVar[RouterFeatures]

    def __init__(self, args: Namespace) -> None:
        self.args = args

    def launch_argv(
        self, ctx: LaunchCommandContext, *, model_cfg: ModelConfig, interpreter_prefix: list[str]
    ) -> list[str]:
        needs_pd_disaggregation = self._needs_pd_disaggregation(model_cfg)
        assert (
            self.features.pd_disaggregation or not needs_pd_disaggregation
        ), f"{self.kind.value} router does not support PD disaggregation."
        return self._launch_argv(ctx, model_cfg=model_cfg, interpreter_prefix=interpreter_prefix)

    def _needs_pd_disaggregation(self, model_cfg: ModelConfig) -> bool:
        return model_cfg.has_pd_disaggregation or self.args.rollout_external_router_pd

    @abstractmethod
    def _launch_argv(
        self, ctx: LaunchCommandContext, *, model_cfg: ModelConfig, interpreter_prefix: list[str]
    ) -> list[str]: ...


class SGLangRouterBackend(RouterBackend):
    kind = RouterKind.SGLANG
    features = RouterFeatures(pd_disaggregation=True)

    def _launch_argv(
        self, ctx: LaunchCommandContext, *, model_cfg: ModelConfig, interpreter_prefix: list[str]
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


class MilesRouterBackend(RouterBackend):
    kind = RouterKind.MILES
    features = RouterFeatures(pd_disaggregation=False)

    def _launch_argv(
        self, ctx: LaunchCommandContext, *, model_cfg: ModelConfig, interpreter_prefix: list[str]
    ) -> list[str]:
        primary = ctx.self_addrs["primary"]
        router_config = compute_miles_router_config(self.args, host=primary.host, port=primary.port)
        return [*interpreter_prefix, "-m", "miles.router.router", *config_to_argv(router_config)]


_BACKEND_CLASSES: dict[RouterKind, type[RouterBackend]] = {
    RouterKind.SGLANG: SGLangRouterBackend,
    RouterKind.MILES: MilesRouterBackend,
}


def make_router_backend(args: Namespace) -> RouterBackend:
    kind = RouterKind.MILES if args.use_miles_router else RouterKind.SGLANG
    return _BACKEND_CLASSES[kind](args)
