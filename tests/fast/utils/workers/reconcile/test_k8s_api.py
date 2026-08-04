from __future__ import annotations

import logging
import sys
from collections.abc import AsyncGenerator
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from miles.utils.workers.reconcile.k8s_api import KubernetesAsyncioPodApi, PodWatchEvent, exception_rejects_cursor


def make_exception(**fields: Any) -> Exception:
    exception = Exception("boom")
    for name, value in fields.items():
        setattr(exception, name, value)
    return exception


class TestKubernetesAsyncioPodApi:
    async def test_stream_pods_swallows_and_logs_a_watch_close_failure(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An exhausted stream logs a watch close failure without exposing it to the caller."""

        class FakeWatch:
            async def stream(self, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
                if False:
                    yield None

            async def close(self) -> None:
                raise RuntimeError("watch close failed")

        watch_module = ModuleType("kubernetes_asyncio.watch")
        watch_module.Watch = FakeWatch
        package = ModuleType("kubernetes_asyncio")
        package.watch = watch_module
        monkeypatch.setitem(sys.modules, "kubernetes_asyncio", package)
        monkeypatch.setitem(sys.modules, "kubernetes_asyncio.watch", watch_module)
        api = KubernetesAsyncioPodApi(core_v1_api=SimpleNamespace(list_namespaced_pod=None))

        with caplog.at_level(logging.ERROR, logger="miles.utils.workers.reconcile.k8s_api"):
            events = [
                event
                async for event in api.stream_pods(
                    namespace="default", label_selector="app=trainer", resource_version="7", timeout_seconds=30
                )
            ]

        assert events == []
        assert caplog.messages == ["failed to close a Kubernetes watch stream"]
        assert caplog.records[0].exc_info is not None


class TestResourceVersionParsing:
    @pytest.mark.parametrize(
        ("obj", "expected"),
        [
            (SimpleNamespace(metadata=SimpleNamespace(resource_version="7")), "7"),
            (dict(metadata=dict(resourceVersion="7")), "7"),
        ],
    )
    @pytest.mark.parametrize("event_type", ["ADDED", "BOOKMARK"])
    def test_both_wire_shapes_are_read(self, event_type: str, obj: Any, expected: str) -> None:
        """A deserialized model spells it as an attribute, a raw dict as a camelCase key."""
        assert PodWatchEvent.from_frame(event_type=event_type, obj=obj).resource_version == expected

    @pytest.mark.parametrize(
        "obj",
        [
            SimpleNamespace(),
            SimpleNamespace(metadata=None),
            SimpleNamespace(metadata=SimpleNamespace()),
            {},
            dict(metadata=None),
            dict(metadata={}),
        ],
    )
    @pytest.mark.parametrize("event_type", ["ADDED", "BOOKMARK"])
    def test_a_frame_without_a_readable_version_parses_to_none(self, event_type: str, obj: Any) -> None:
        """A missing or malformed metadata block must parse to None, never raise: the caller keeps its cursor."""
        assert PodWatchEvent.from_frame(event_type=event_type, obj=obj).resource_version is None

    @pytest.mark.parametrize("event_type", ["ADDED", "BOOKMARK"])
    def test_a_non_object_payload_parses_to_none(self, event_type: str) -> None:
        """A non-object payload has no readable resource version in either frame type."""
        event = PodWatchEvent.from_frame(event_type=event_type, obj="a payload that is not an object at all")

        assert event.resource_version is None
        assert event.obj == "a payload that is not an object at all"


class TestCursorRejection:
    @pytest.mark.parametrize(
        "obj",
        [
            SimpleNamespace(code=410, reason="Expired"),
            SimpleNamespace(code=504, reason="Timeout"),
            SimpleNamespace(code=None, reason="Expired"),
            dict(code=410),
            dict(reason="Gone"),
        ],
    )
    def test_an_error_frame_reporting_a_dead_cursor_is_flagged(self, obj: Any) -> None:
        """Either the code or the reason is enough, in either wire shape."""
        assert PodWatchEvent.from_frame(event_type="ERROR", obj=obj).rejects_cursor

    @pytest.mark.parametrize(
        ("event_type", "obj"),
        [
            ("ERROR", SimpleNamespace(code=500, reason="InternalError")),
            ("ERROR", dict(code=500)),
            ("ERROR", SimpleNamespace()),
            ("MODIFIED", SimpleNamespace(code=410, reason="Expired")),
            ("BOOKMARK", dict(code=410)),
        ],
    )
    def test_anything_else_leaves_the_cursor_alone(self, event_type: str, obj: Any) -> None:
        """Only an ERROR frame may invalidate a cursor, and only for a cursor-specific code."""
        assert not PodWatchEvent.from_frame(event_type=event_type, obj=obj).rejects_cursor

    def test_a_pod_frame_carrying_a_dead_cursor_code_leaves_the_cursor_alone(self) -> None:
        """A pod whose own fields happen to spell 410 must not be read as an expired-cursor error."""
        obj = SimpleNamespace(metadata=SimpleNamespace(name="pod-0", resource_version="7"))
        obj.code = 410
        obj.reason = "Expired"

        event = PodWatchEvent.from_frame(event_type="MODIFIED", obj=obj)

        assert not event.rejects_cursor
        assert event.obj is obj
        assert event.obj.metadata.name == "pod-0"


class TestExceptionRejection:
    @pytest.mark.parametrize(
        "exception",
        [make_exception(status=410), make_exception(status=504)],
    )
    def test_a_client_exception_reporting_a_dead_cursor_is_flagged(self, exception: BaseException) -> None:
        """ApiException.status carries the HTTP code, and 410 and 504 both mean the cursor is gone."""
        assert exception_rejects_cursor(exception)

    @pytest.mark.parametrize(
        "exception",
        [make_exception(), make_exception(status=500), make_exception(status="410")],
    )
    def test_any_other_exception_is_a_plain_stream_failure(self, exception: BaseException) -> None:
        """A transient error, or a code in a shape the client never produces, keeps the cursor."""
        assert not exception_rejects_cursor(exception)
