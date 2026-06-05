"""Dynamo-backed inference helpers for Miles rollouts.

Two pieces:

* :mod:`.dynamo_engine` — a Ray actor that owns one ``dynamo.sglang`` worker
  subprocess. Public method surface matches :class:`SGLangEngine` so it slots
  into :class:`ServerGroup` without further plumbing.

* :mod:`.dynamo_router` — replaces ``_start_router``. Launches a
  ``dynamo.frontend`` subprocess (and, if KV routing is requested, also etcd
  + NATS). Returns ``(ip, port)`` so the rest of Miles continues to talk to
  ``args.sglang_router_ip/port``.
"""
