"""Per-arch adaptation layer for the FSDP backend.

``arch_adapter.ArchAdapter`` declares every hook the actor runs around stock HF modeling; ``specs/`` holds
one adapter per architecture and ``specs.resolve_arch_adapter`` picks one by ``model_type``. The other
modules here are the mechanisms those hooks share: ``packing`` (document boundaries), ``precision``,
``routing_replay``, ``weight_bridge`` and ``config_checks``.
"""
