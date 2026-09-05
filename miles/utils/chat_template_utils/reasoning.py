from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from miles.utils.arg_resolution import (
    MISSING,
    ArgBatch,
    ArgResolutionContract,
    ArgResolutionError,
    Binding,
    NonePolicy,
    PrimaryField,
    PrimarySchema,
    SourceSpec,
)


def _active_effort(value):
    return MISSING if value == "none" else value


def _launch_toggle(value):
    return False if value == "none" else MISSING


def _request_toggle(value):
    return value != "none"


def _object_toggle(value):
    if isinstance(value, str):
        value = value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return True if value else MISSING


@dataclass(frozen=True)
class ReasoningTemplateConfig:
    toggle_kwarg: str
    default_effort: str
    active_efforts: frozenset[str]
    resolver: ArgResolutionContract = field(init=False, repr=False)

    def __post_init__(self):
        active_efforts = frozenset(self.active_efforts)
        if self.default_effort not in active_efforts:
            raise ValueError("default_effort must be one of active_efforts")
        object.__setattr__(self, "active_efforts", active_efforts)
        effort = Binding(
            "reasoning_effort", "reasoning_effort", none_policy=NonePolicy.MISSING, project=_active_effort
        )
        request_toggle = Binding(
            "reasoning_effort", self.toggle_kwarg, none_policy=NonePolicy.MISSING, project=_request_toggle
        )
        resolver = ArgResolutionContract(
            PrimarySchema((PrimaryField("reasoning_effort"), PrimaryField(self.toggle_kwarg))),
            (
                SourceSpec("family", 0, (effort,)),
                SourceSpec(
                    "launch",
                    10,
                    (
                        effort,
                        Binding(
                            "reasoning_effort",
                            self.toggle_kwarg,
                            none_policy=NonePolicy.MISSING,
                            project=_launch_toggle,
                        ),
                        Binding(self.toggle_kwarg, self.toggle_kwarg, priority=11),
                    ),
                ),
                SourceSpec("top_level", 20, (effort,)),
                SourceSpec("object_enabled", 30, (Binding("enabled", self.toggle_kwarg, project=_object_toggle),)),
                SourceSpec("selected_effort", 40, (effort, request_toggle)),
                SourceSpec("nested_effort", 50, (effort, request_toggle)),
                SourceSpec("nested_toggle", 60, (Binding(self.toggle_kwarg, self.toggle_kwarg),)),
            ),
        )
        object.__setattr__(self, "resolver", resolver)

    def resolve(self, launch_kwargs: Mapping[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
        nested = request.get("chat_template_kwargs") or {}
        reasoning = request.get("reasoning")
        reasoning = reasoning if isinstance(reasoning, dict) else {}
        raw_efforts = (
            ("launch.reasoning_effort", launch_kwargs.get("reasoning_effort")),
            ("reasoning_effort", request.get("reasoning_effort")),
            ("reasoning.effort", reasoning.get("effort")),
            ("reasoning.reasoning_effort", reasoning.get("reasoning_effort")),
            ("chat_template_kwargs.reasoning_effort", nested.get("reasoning_effort")),
        )
        for name, value in raw_efforts:
            if value is not None and (not isinstance(value, str) or value not in self.active_efforts | {"none"}):
                raise ArgResolutionError(f"Invalid reasoning effort for {name}: {value!r}")

        object_effort = reasoning.get("effort")
        if object_effort is None:
            object_effort = reasoning.get("reasoning_effort")
        selected_effort = request.get("reasoning_effort") if object_effort is None else object_effort
        enabled = reasoning.get("enabled")
        if enabled is None:
            enabled = reasoning.get("enable")
        resolved = self.resolver.resolve(
            (
                ArgBatch("family", {"reasoning_effort": self.default_effort}),
                ArgBatch(
                    "launch",
                    {
                        key: launch_kwargs[key]
                        for key in ("reasoning_effort", self.toggle_kwarg)
                        if key in launch_kwargs
                    },
                ),
                ArgBatch("top_level", {"reasoning_effort": request.get("reasoning_effort")}),
                ArgBatch("object_enabled", {"enabled": enabled}),
                ArgBatch("selected_effort", {"reasoning_effort": selected_effort}),
                ArgBatch("nested_effort", {"reasoning_effort": nested.get("reasoning_effort")}),
                ArgBatch(
                    "nested_toggle",
                    {self.toggle_kwarg: nested[self.toggle_kwarg]} if self.toggle_kwarg in nested else {},
                ),
            )
        )
        kwargs = {**launch_kwargs, **nested}
        kwargs.pop("reasoning_effort", None)
        kwargs.pop(self.toggle_kwarg, None)
        kwargs.update(resolved.values)
        return kwargs


QWEN38_REASONING = ReasoningTemplateConfig(
    toggle_kwarg="enable_thinking",
    default_effort="xhigh",
    active_efforts=frozenset({"low", "medium", "xhigh"}),
)
