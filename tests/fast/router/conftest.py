"""Shared fixtures for router tests.

The class-scoped ``router_env`` (mock SGLang upstream + session server) is
defined in test_sessions.py; re-exported here so additive test modules can
depend on it without importing a name their own test signatures would shadow.
"""

from tests.fast.router.test_sessions import router_env  # noqa: F401
