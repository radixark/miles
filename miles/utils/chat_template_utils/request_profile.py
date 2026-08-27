from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RequestIntent:
    """Canonical renderer inputs extracted from one client request."""

    chat_template_kwargs: Mapping[str, Any] = field(default_factory=dict)
    chat_template_kwargs_present: bool = False
    consumed_fields: frozenset[str] = frozenset()


class ModelRequestProfile:
    """Translate model-specific request syntax into canonical renderer inputs."""

    def extract(self, request_body: Mapping[str, Any]) -> RequestIntent:
        request_kwargs = request_body.get("chat_template_kwargs")
        if request_kwargs is None:
            return RequestIntent(consumed_fields=frozenset({"chat_template_kwargs"}))
        if not isinstance(request_kwargs, dict):
            raise ValueError("chat_template_kwargs must be an object")
        return RequestIntent(
            chat_template_kwargs=dict(request_kwargs),
            chat_template_kwargs_present=True,
            consumed_fields=frozenset({"chat_template_kwargs"}),
        )

    def render_fingerprint(self, render_kwargs: Mapping[str, Any]) -> Hashable | None:
        """Return the prefix-sensitive renderer identity, if this profile has one."""
        return None


DEFAULT_REQUEST_PROFILE = ModelRequestProfile()
