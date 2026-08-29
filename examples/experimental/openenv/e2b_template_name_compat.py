"""Send the template name under the field a self-hosted server expects.

The e2b Python SDK (every release through 2.10.2) submits a template build as
``POST /v3/templates`` with the name in ``alias``. AgentENV's endpoint reads
``name`` and rejects a body without it (radixark/AgentENV#8, fix proposed
upstream). Until a server that accepts ``alias`` is deployed, this copies the
value into the field named by OPENENV_E2B_TEMPLATE_NAME_FIELD before the
request is serialized. The SDK's own field is kept, so a server that reads
either works.

Patching the request model is the narrowest hook: the SDK builds the body in
its generated client, below anything the caller can pass through
``Template.build``. Remove this module once the deployed AgentENV reads
``alias``.
"""

import os

ENV_VAR = "OPENENV_E2B_TEMPLATE_NAME_FIELD"
_APPLIED_ATTR = "_openenv_template_name_field"


def apply_from_env() -> str | None:
    """Install the shim if OPENENV_E2B_TEMPLATE_NAME_FIELD is set; return the
    field name it maps to, or None when the variable is unset or empty."""
    field = os.getenv(ENV_VAR, "").strip()
    if not field:
        return None
    apply(field)
    return field


def apply(field: str) -> None:
    """Make TemplateBuildRequestV3 serialize its alias under *field* too.
    Idempotent: a second call with the same field is a no-op; a different field
    replaces the mapping."""
    # In-function import: the e2b SDK is an optional dependency of this recipe,
    # and the module must import without it (the offline tests fake `e2b`).
    from e2b.api.client.models import template_build_request_v3 as request_model

    cls = request_model.TemplateBuildRequestV3
    original = getattr(cls, "_openenv_original_to_dict", None) or cls.to_dict

    def to_dict(self):
        body = original(self)
        if "alias" in body and field not in body:
            body[field] = body["alias"]
        return body

    cls._openenv_original_to_dict = original
    cls.to_dict = to_dict
    setattr(cls, _APPLIED_ATTR, field)
