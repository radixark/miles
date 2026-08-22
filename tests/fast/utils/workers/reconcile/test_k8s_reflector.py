from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import AsyncIterator
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from tests.fast.utils.workers.reconcile.utils import (
    EventCollector,
    FakeApiException,
    FakePodApi,
    ParentKeyRecorder,
    make_pod,
    make_pod_list,
    pod_cell,
    settle,
    wire_pod,
)

from miles.utils.test_utils.clock import FakeClock
from miles.utils.workers.reconcile.k8s_api import KubernetesAsyncioPodApi, PodListPage, PodWatchEvent
from miles.utils.workers.reconcile.k8s_reflector import KubernetesReflector
from miles.utils.workers.reconcile.loop import ReconcileLoop
from miles.utils.workers.reconcile.source_event import DeleteEvent, ReplaceEvent, UpsertEvent


def raw_event(event_type: str, obj: Any) -> PodWatchEvent:
    return PodWatchEvent.from_frame(event_type=event_type, obj=obj)


def _an_unreadable_frame_envelope() -> ValidationError:
    try:
        PodWatchEvent.from_frame(event_type="BOOKMARK", obj="not an object")
    except ValidationError as error:
        return error
    raise AssertionError("the envelope parsed, so it no longer stands for one the parser rejects")


def make_status(*, code: int, reason: str = "Expired") -> SimpleNamespace:
    return SimpleNamespace(code=code, reason=reason, status="Failure")


def _install_fake_kubernetes_asyncio(monkeypatch: Any) -> tuple[Any, dict[str, Any]]:
    state: dict[str, Any] = dict(func=None, kwargs=None, closed=0)

    class _FakeWatch:
        def stream(self, func: Any, **kwargs: Any) -> Any:
            state["func"] = func
            state["kwargs"] = kwargs
            return self

        def __aiter__(self) -> Any:
            async def _iterate() -> AsyncIterator[dict[str, Any]]:
                yield dict(type="MODIFIED", object=wire_pod("pod-from-the-wire"))

            return _iterate()

        async def close(self) -> None:
            state["closed"] += 1

    watch_module = ModuleType("kubernetes_asyncio.watch")
    watch_module.Watch = _FakeWatch
    package = ModuleType("kubernetes_asyncio")
    package.watch = watch_module
    monkeypatch.setitem(sys.modules, "kubernetes_asyncio", package)
    monkeypatch.setitem(sys.modules, "kubernetes_asyncio.watch", watch_module)
    return watch_module, state


def make_reflector(api: FakePodApi, *, clock: FakeClock | None = None, **kwargs: Any) -> KubernetesReflector:
    return KubernetesReflector(
        kube_client=api,
        namespace="ns-test",
        label_selector="app=miles",
        clock=clock or FakeClock(),
        **kwargs,
    )


class TestCadenceValidation:
    @pytest.mark.parametrize("kwargs", [dict(retry_delay=0.0), dict(retry_delay=-1.0), dict(watch_timeout_seconds=0)])
    async def test_a_non_positive_cadence_is_rejected(self, kwargs):
        """A zero retry delay or watch timeout would hammer the apiserver, so it never reaches construction."""
        with pytest.raises(AssertionError):
            make_reflector(FakePodApi(), **kwargs)


class TestInitialList:
    async def test_list_emits_one_replace_carrying_every_pod(self):
        """The initial LIST arrives as a single whole-world ReplaceEvent, keyed by pod name."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([make_pod("pod-0"), make_pod("pod-1")], resource_version="100"))
        collector = EventCollector(make_reflector(api).watch())
        await settle()

        assert [type(event) for event in collector.events] == [ReplaceEvent]
        assert sorted(collector.events[0].objects) == ["pod-0", "pod-1"]
        await collector.close()

    async def test_list_uses_namespace_and_label_selector(self):
        """LIST is scoped to the configured namespace and selector."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        collector = EventCollector(make_reflector(api).watch())
        await settle()

        assert api.list_calls == [dict(namespace="ns-test", label_selector="app=miles")]
        await collector.close()

    async def test_watch_starts_from_the_list_resource_version(self):
        """WATCH continues exactly where the LIST snapshot ended."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="100"))
        collector = EventCollector(make_reflector(api, watch_timeout_seconds=42).watch())
        await settle()

        assert api.stream_calls == [
            dict(namespace="ns-test", label_selector="app=miles", resource_version="100", timeout_seconds=42)
        ]
        await collector.close()

    async def test_the_default_watch_timeout_is_five_minutes(self):
        """A caller that configures nothing still gets a finite server-side watch timeout."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        collector = EventCollector(make_reflector(api).watch())
        await settle()

        assert api.stream_calls[0]["timeout_seconds"] == 300
        await collector.close()

    async def test_empty_list_still_emits_a_replace(self):
        """An empty cluster must still lift the initial-sync barrier."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        collector = EventCollector(make_reflector(api).watch())
        await settle()

        assert [type(event) for event in collector.events] == [ReplaceEvent]
        await collector.close()

    async def test_list_failure_is_retried_after_the_delay(self):
        """A failing LIST is retried; no events escape before it succeeds."""
        api = FakePodApi()
        api.list_pages.append(RuntimeError("apiserver down"))
        api.list_pages.append(make_pod_list([make_pod("pod-0")], resource_version="7"))
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=3.0).watch())
        await settle()
        assert collector.events == []

        await clock.elapse(3.0)
        await settle()
        assert [type(event) for event in collector.events] == [ReplaceEvent]
        await collector.close()


class TestWatchEvents:
    async def test_added_and_modified_map_to_upsert(self):
        """ADDED and MODIFIED both become UpsertEvent."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        api.stream_scripts.append(
            [
                raw_event("ADDED", make_pod("pod-0", resource_version="2")),
                raw_event("MODIFIED", make_pod("pod-0", resource_version="3")),
            ]
        )
        collector = EventCollector(make_reflector(api).watch())
        await settle()

        assert collector.events[1:] == [
            UpsertEvent(key="pod-0", obj=collector.events[1].obj),
            UpsertEvent(key="pod-0", obj=collector.events[2].obj),
        ]
        assert collector.events[2].obj.metadata.resource_version == "3"
        await collector.close()

    async def test_deleted_maps_to_delete_with_tombstone(self):
        """DELETED carries the last known object so the consumer can attribute it."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        pod = make_pod("pod-0", cell="cell-x", resource_version="5")
        api.stream_scripts.append([raw_event("DELETED", pod)])
        collector = EventCollector(make_reflector(api).watch())
        await settle()

        assert collector.events[1] == DeleteEvent(key="pod-0", last_obj=pod)
        await collector.close()

    async def test_bookmark_emits_nothing_but_advances_the_cursor(self):
        """BOOKMARK only refreshes the resource version."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        api.stream_scripts.append([raw_event("BOOKMARK", make_pod("pod-ignored", resource_version="50"))])
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=1.0).watch())
        await settle()
        await clock.elapse(1.0)
        await settle()

        assert [type(event) for event in collector.events] == [ReplaceEvent]
        assert api.stream_calls[-1]["resource_version"] == "50"
        assert len(api.list_calls) == 1
        await collector.close()

    async def test_raw_dict_bookmark_advances_the_cursor(self):
        """kubernetes_asyncio leaves BOOKMARK payloads as raw dicts; the cursor must still move."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        api.stream_scripts.append([raw_event("BOOKMARK", dict(metadata=dict(resourceVersion="77")))])
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=1.0).watch())
        await settle()
        await clock.elapse(1.0)
        await settle()

        assert api.stream_calls[-1]["resource_version"] == "77"
        assert len(api.list_calls) == 1
        await collector.close()

    async def test_unknown_event_type_is_ignored(self):
        """An unrecognized event type does not break the stream."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        api.stream_scripts.append(
            [raw_event("WEIRD", make_pod("pod-0")), raw_event("ADDED", make_pod("pod-1", resource_version="2"))]
        )
        api.stream_scripts.append(None)
        collector = EventCollector(make_reflector(api).watch())
        await settle()

        assert [type(event) for event in collector.events] == [ReplaceEvent, UpsertEvent]
        assert collector.events[1].key == "pod-1"
        await collector.close()

    @pytest.mark.parametrize("event_type", ["ADDED", "MODIFIED", "DELETED"])
    async def test_a_pod_event_carrying_no_pod_fails_the_stream(self, event_type, caplog):
        """Skipping it would leave the store silently short of a pod, so the stream must fail instead."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        api.stream_scripts.append(
            [PodWatchEvent(type=event_type, pod=None, resource_version="5", rejects_cursor=False)]
        )
        api.stream_scripts.append(None)
        collector = EventCollector(make_reflector(api).watch())
        with caplog.at_level(logging.ERROR, logger="miles.utils.workers.reconcile.k8s_reflector"):
            await settle()

        assert [type(event) for event in collector.events] == [ReplaceEvent]
        assert "carries no pod" in caplog.text
        await collector.close()

    async def test_a_stream_that_failed_relists_rather_than_replaying_the_frame_that_failed_it(self):
        """The cursor never advanced past the poison frame, so resuming from it would redeliver it forever.

        A transient stream error is different and keeps its cursor: see
        test_transient_watch_error_retries_without_relisting."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        api.list_pages.append(make_pod_list([], resource_version="9"))
        api.stream_scripts.append(
            [PodWatchEvent(type="MODIFIED", pod=None, resource_version="5", rejects_cursor=False)]
        )
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=1.0).watch())
        await settle()
        await clock.elapse(1.0)
        await settle()

        assert len(api.list_calls) == 2
        assert [call["resource_version"] for call in api.stream_calls] == ["1", "9"]
        await collector.close()

    async def test_a_malformed_pod_event_is_skipped_and_advances_the_cursor(self, caplog):
        """A malformed pod payload must not block later events or be replayed after the watch reopens."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        api.stream_scripts.append(
            [
                PodWatchEvent(
                    type="MODIFIED",
                    pod=None,
                    resource_version="5",
                    rejects_cursor=False,
                    pod_validation_error="restart_count is missing",
                ),
                raw_event("ADDED", make_pod("pod-0", resource_version="6")),
            ]
        )
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=1.0).watch())
        with caplog.at_level(logging.ERROR, logger="miles.utils.workers.reconcile.k8s_reflector"):
            await settle()
            await clock.elapse(1.0)
            await settle()

        assert [type(event) for event in collector.events] == [ReplaceEvent, UpsertEvent]
        assert collector.events[1].key == "pod-0"
        assert len(api.list_calls) == 1
        assert [call["resource_version"] for call in api.stream_calls] == ["1", "6"]
        assert "skipping a malformed pod watch event" in caplog.text
        await collector.close()

    async def test_a_malformed_pod_event_without_a_cursor_relists(self):
        """A malformed payload without a resource version cannot be skipped without risking replay."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        api.list_pages.append(make_pod_list([], resource_version="9"))
        api.stream_scripts.append(
            [
                PodWatchEvent(
                    type="MODIFIED",
                    pod=None,
                    resource_version=None,
                    rejects_cursor=False,
                    pod_validation_error="metadata.resourceVersion is missing",
                )
            ]
        )
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=1.0).watch())
        await settle()
        await clock.elapse(1.0)
        await settle()

        assert len(api.list_calls) == 2
        assert [call["resource_version"] for call in api.stream_calls] == ["1", "9"]
        await collector.close()

    async def test_an_unreadable_frame_envelope_relists(self):
        """An envelope that exposes no cursor cannot be skipped safely, so it rebuilds from LIST."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        api.list_pages.append(make_pod_list([], resource_version="9"))
        api.stream_scripts.append([_an_unreadable_frame_envelope()])
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=1.0).watch())
        await settle()
        await clock.elapse(1.0)
        await settle()

        assert len(api.list_calls) == 2
        assert [call["resource_version"] for call in api.stream_calls] == ["1", "9"]
        await collector.close()

    async def test_watch_end_resumes_without_relisting(self):
        """A watch that ends cleanly is reopened from the latest cursor, with no second ReplaceEvent."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        api.stream_scripts.append([raw_event("ADDED", make_pod("pod-0", resource_version="9"))])
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=1.0).watch())
        await settle()
        assert [call["resource_version"] for call in api.stream_calls] == ["1"]

        await clock.elapse(1.0)
        await settle()
        assert [type(event) for event in collector.events] == [ReplaceEvent, UpsertEvent]
        assert len(api.list_calls) == 1
        assert [call["resource_version"] for call in api.stream_calls] == ["1", "9"]
        await collector.close()

    async def test_a_list_that_cannot_be_converted_does_not_advance_the_cursor(self):
        """A LIST that cannot be converted is retried as a LIST, not silently downgraded to a WATCH."""
        api = FakePodApi()
        api.list_pages.append(RuntimeError("a listed pod could not be converted"))
        api.list_pages.append(make_pod_list([make_pod("pod-0")], resource_version="2"))
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=1.0).watch())
        await settle()
        assert collector.events == []

        await clock.elapse(1.0)
        await settle()
        assert [type(event) for event in collector.events] == [ReplaceEvent]
        assert len(api.list_calls) == 2
        assert api.stream_calls[-1]["resource_version"] == "2"
        await collector.close()

    async def test_transient_watch_error_retries_without_relisting(self):
        """A non-410 stream error backs off and resumes from the last cursor."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        api.stream_scripts.append(
            [raw_event("ADDED", make_pod("pod-0", resource_version="4")), RuntimeError("connection reset")]
        )
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=2.0).watch())
        await settle()
        assert len(api.stream_calls) == 1

        await clock.elapse(2.0)
        await settle()
        assert len(api.list_calls) == 1
        assert [call["resource_version"] for call in api.stream_calls] == ["1", "4"]
        await collector.close()

    async def test_the_reconnect_cursor_follows_event_order_not_version_comparison(self):
        """Resource versions are opaque strings: the newest event wins even when its value looks smaller."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        api.stream_scripts.append(
            [
                raw_event("ADDED", make_pod("pod-0", resource_version="55")),
                raw_event("MODIFIED", make_pod("pod-0", resource_version="32")),
            ]
        )
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=1.0).watch())
        await settle()

        await clock.elapse(1.0)
        await settle()
        assert [call["resource_version"] for call in api.stream_calls] == ["1", "32"]
        await collector.close()

    async def test_a_non_expiry_error_event_closes_the_stream_before_reconnecting(self):
        """A raw ERROR frame that is not an expiry tears its stream down instead of leaking it."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        api.stream_scripts.append(
            [
                raw_event("ERROR", make_status(code=500, reason="InternalError")),
                raw_event("ADDED", make_pod("pod-0", resource_version="9")),
            ]
        )
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=2.0).watch())
        await settle()

        assert api.closed_streams == 1
        assert len(api.list_calls) == 1
        assert [type(event) for event in collector.events] == [ReplaceEvent]

        await clock.elapse(2.0)
        await settle()
        assert [call["resource_version"] for call in api.stream_calls] == ["1", "1"]
        await collector.close()

    async def test_streams_are_closed_on_reopen(self):
        """Every finished watch stream is closed before the next is opened."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        api.stream_scripts.append([])
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=1.0).watch())
        await settle()

        assert api.closed_streams == 1
        assert len(api.stream_calls) == 1

        await clock.elapse(1.0)
        await settle()

        assert len(api.stream_calls) == 2
        await collector.close()


class TestExpiry:
    async def test_expired_error_event_triggers_a_relist(self):
        """A 410 ERROR event forces a fresh LIST and a new whole-world ReplaceEvent."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([make_pod("pod-0")], resource_version="1"))
        api.stream_scripts.append([raw_event("ERROR", make_status(code=410))])
        api.list_pages.append(make_pod_list([make_pod("pod-1")], resource_version="200"))
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=1.0).watch())
        await settle()
        await clock.elapse(1.0)
        await settle()

        assert [type(event) for event in collector.events] == [ReplaceEvent, ReplaceEvent]
        assert [sorted(event.objects) for event in collector.events] == [["pod-0"], ["pod-1"]]
        assert len(api.list_calls) == 2
        assert api.stream_calls[-1]["resource_version"] == "200"
        await collector.close()

    async def test_an_expired_error_event_waits_for_the_retry_delay_before_relisting(self):
        """410 gets no free instant relist: it is paced like every other LIST/WATCH cycle."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        api.stream_scripts.append([raw_event("ERROR", make_status(code=410))])
        api.list_pages.append(make_pod_list([], resource_version="200"))
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=3.0).watch())
        await settle()
        assert len(api.list_calls) == 1

        await clock.elapse(2.0)
        await settle()
        assert len(api.list_calls) == 1

        await clock.elapse(1.0)
        await settle()
        assert len(api.list_calls) == 2
        await collector.close()

    async def test_expired_error_event_as_dict_triggers_a_relist(self):
        """The raw dict form of a 410 status is recognized too."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        api.stream_scripts.append([raw_event("ERROR", dict(code=410, reason="Expired"))])
        api.list_pages.append(make_pod_list([], resource_version="2"))
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=1.0).watch())
        await settle()
        await clock.elapse(1.0)
        await settle()

        assert [type(event) for event in collector.events] == [ReplaceEvent, ReplaceEvent]
        assert len(api.list_calls) == 2
        await collector.close()

    async def test_expired_api_exception_triggers_a_relist(self):
        """An ApiException(410) raised by the client forces a relist."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        api.stream_scripts.append([FakeApiException(410)])
        api.list_pages.append(make_pod_list([], resource_version="300"))
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=5.0).watch())
        await settle()
        assert len(api.list_calls) == 1

        await clock.elapse(5.0)
        await settle()

        assert [type(event) for event in collector.events] == [ReplaceEvent, ReplaceEvent]
        assert api.stream_calls[-1]["resource_version"] == "300"
        await collector.close()

    async def test_a_too_large_resource_version_exception_triggers_a_relist(self):
        """A cursor from the future is unusable, so it is dropped for a fresh LIST instead of retried forever."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="5000"))
        api.stream_scripts.append([FakeApiException(504)])
        api.list_pages.append(make_pod_list([], resource_version="100"))
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=1.0).watch())
        await settle()
        await clock.elapse(1.0)
        await settle()

        assert [type(event) for event in collector.events] == [ReplaceEvent, ReplaceEvent]
        assert [call["resource_version"] for call in api.stream_calls] == ["5000", "100"]
        await collector.close()

    async def test_a_too_large_resource_version_error_event_triggers_a_relist(self):
        """The apiserver may report the same condition as an ERROR frame rather than a status exception."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="5000"))
        api.stream_scripts.append([raw_event("ERROR", make_status(code=504, reason="Timeout"))])
        api.list_pages.append(make_pod_list([], resource_version="100"))
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=1.0).watch())
        await settle()
        await clock.elapse(1.0)
        await settle()

        assert len(api.list_calls) == 2
        assert [call["resource_version"] for call in api.stream_calls] == ["5000", "100"]
        await collector.close()

    async def test_non_expiry_error_event_is_a_stream_failure(self):
        """A non-410 ERROR event is treated as a failure, not as an expiry."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        api.stream_scripts.append([raw_event("ERROR", make_status(code=500, reason="InternalError"))])
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=2.0).watch())
        await settle()
        assert len(api.list_calls) == 1

        await clock.elapse(2.0)
        await settle()
        assert len(api.list_calls) == 1
        assert [call["resource_version"] for call in api.stream_calls] == ["1", "1"]
        await collector.close()

    async def test_relist_replays_the_full_world(self):
        """After expiry the consumer receives the complete current state, not a delta."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([make_pod("pod-0"), make_pod("pod-1")], resource_version="1"))
        api.stream_scripts.append([FakeApiException(410)])
        api.list_pages.append(make_pod_list([make_pod("pod-1"), make_pod("pod-2")], resource_version="2"))
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=1.0).watch())
        await settle()
        await clock.elapse(1.0)
        await settle()

        assert sorted(collector.events[1].objects) == ["pod-1", "pod-2"]
        await collector.close()


class TestCursor:
    @pytest.mark.parametrize("event_type", ["MODIFIED", "DELETED"])
    async def test_every_event_kind_advances_the_reconnect_cursor(self, event_type):
        """The cursor follows the newest observed object, whatever the event type."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        api.stream_scripts.append([raw_event(event_type, make_pod("pod-0", resource_version="7"))])
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=1.0).watch())
        await settle()
        await clock.elapse(1.0)
        await settle()

        assert [call["resource_version"] for call in api.stream_calls] == ["1", "7"]
        await collector.close()

    async def test_event_without_resource_version_keeps_the_previous_cursor(self):
        """An object with no metadata must not reset the cursor."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="11"))
        api.stream_scripts.append([raw_event("BOOKMARK", dict(other="thing"))])
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=1.0).watch())
        await settle()
        await clock.elapse(1.0)
        await settle()

        assert [call["resource_version"] for call in api.stream_calls] == ["11", "11"]
        assert len(api.list_calls) == 1
        await collector.close()

    async def test_expiry_discards_the_rest_of_the_stale_stream(self):
        """Events queued behind a 410 belong to a dead cursor and must not be delivered."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        api.stream_scripts.append(
            [raw_event("ERROR", make_status(code=410)), raw_event("ADDED", make_pod("pod-stale"))]
        )
        api.list_pages.append(make_pod_list([], resource_version="2"))
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=1.0).watch())
        await settle()
        await clock.elapse(1.0)
        await settle()

        assert [type(event) for event in collector.events] == [ReplaceEvent, ReplaceEvent]
        await collector.close()

    async def test_watch_after_a_relist_reconnects_without_listing_again(self):
        """The expired flag is cleared by the relist, so the next clean end is an ordinary reconnect."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        api.stream_scripts.append([FakeApiException(410)])
        api.list_pages.append(make_pod_list([], resource_version="100"))
        api.stream_scripts.append([raw_event("ADDED", make_pod("pod-0", resource_version="101"))])
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=1.0).watch())
        await settle()
        await clock.elapse(1.0)
        await settle()
        await clock.elapse(1.0)
        await settle()

        assert len(api.list_calls) == 2
        assert [call["resource_version"] for call in api.stream_calls] == ["1", "100", "101"]
        await collector.close()


class TestStreamClosing:
    async def test_an_expired_stream_is_closed_before_the_relist(self):
        """Breaking out on 410 must actively close the watch, not wait for GC."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        api.stream_scripts.append([raw_event("ERROR", make_status(code=410)), raw_event("ADDED", make_pod("pod-0"))])
        api.list_pages.append(make_pod_list([], resource_version="2"))
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=1.0).watch())
        await settle()

        assert api.aclosed_streams == 1
        await collector.close()

    async def test_a_cancelled_watch_leaves_no_open_stream(self):
        """Cancelling the consumer must not leave the watch open."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        api.stream_scripts.append(None)
        collector = EventCollector(make_reflector(api).watch())
        await settle()
        assert api.closed_streams == 0

        await collector.close()
        await settle()
        assert api.closed_streams == 1


class TestLifecycle:
    async def test_cancellation_stops_the_generator(self):
        """Cancelling the consumer tears the reflector down without error."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        collector = EventCollector(make_reflector(api).watch())
        await settle()
        await collector.close()

        assert collector.error is None

    async def test_each_watch_call_is_an_independent_stream(self):
        """watch() can be called again to get a fresh LIST-then-WATCH stream."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([make_pod("pod-0")], resource_version="1"))
        api.stream_scripts.append(None)
        api.list_pages.append(make_pod_list([make_pod("pod-0")], resource_version="2"))
        api.stream_scripts.append(None)
        reflector = make_reflector(api)

        first = EventCollector(reflector.watch())
        await settle()
        await first.close()
        second = EventCollector(reflector.watch())
        await settle()

        assert [type(event) for event in second.events] == [ReplaceEvent]
        assert [call["resource_version"] for call in api.stream_calls] == ["1", "2"]
        await second.close()


class TestKubernetesAsyncioPodApi:
    async def test_list_pods_delegates_to_core_v1_api(self):
        """The adapter forwards LIST to CoreV1Api.list_namespaced_pod."""
        calls: list[dict[str, Any]] = []

        class _CoreV1Api:
            async def list_namespaced_pod(self, **kwargs: Any) -> Any:
                calls.append(kwargs)
                return SimpleNamespace(items=[wire_pod("pod-0")], metadata=SimpleNamespace(resource_version="100"))

        api = KubernetesAsyncioPodApi(core_v1_api=_CoreV1Api())
        page = await api.list_pods(namespace="ns", label_selector="a=b")

        assert page == PodListPage(pods=[make_pod("pod-0")], resource_version="100")
        assert calls == [dict(namespace="ns", label_selector="a=b")]

    async def test_list_pods_refuses_an_item_that_is_not_a_pod(self):
        """A page miles cannot read must fail the LIST rather than reach the store half-converted."""

        class _CoreV1Api:
            async def list_namespaced_pod(self, **kwargs: Any) -> Any:
                return SimpleNamespace(items=["not a pod"], metadata=SimpleNamespace(resource_version="100"))

        api = KubernetesAsyncioPodApi(core_v1_api=_CoreV1Api())

        with pytest.raises(ValidationError):
            await api.list_pods(namespace="ns", label_selector="a=b")

    async def test_stream_pods_forwards_watch_options_and_closes_the_watch(self, monkeypatch):
        """The adapter drives kubernetes_asyncio's Watch and closes it, which has close() and no aclose()."""
        watch_module, state = _install_fake_kubernetes_asyncio(monkeypatch)
        _ = watch_module
        list_namespaced_pod = object()

        class _CoreV1Api:
            pass

        core_v1_api = _CoreV1Api()
        core_v1_api.list_namespaced_pod = list_namespaced_pod
        api = KubernetesAsyncioPodApi(core_v1_api=core_v1_api)

        events = []
        async for event in api.stream_pods(
            namespace="ns", label_selector="a=b", resource_version="42", timeout_seconds=7
        ):
            events.append(event)

        assert len(events) == 1
        assert events[0].type == "MODIFIED"
        assert events[0].pod == make_pod("pod-from-the-wire")
        assert events[0].rejects_cursor is False
        assert state["func"] is list_namespaced_pod
        assert state["kwargs"] == dict(
            namespace="ns",
            label_selector="a=b",
            resource_version="42",
            timeout_seconds=7,
            allow_watch_bookmarks=True,
        )
        assert state["closed"] == 1

    async def test_stream_pods_closes_the_watch_when_the_consumer_stops_early(self, monkeypatch):
        """Abandoning the stream must not leak the underlying watch connection."""
        _, state = _install_fake_kubernetes_asyncio(monkeypatch)

        class _CoreV1Api:
            pass

        core_v1_api = _CoreV1Api()
        core_v1_api.list_namespaced_pod = object()
        api = KubernetesAsyncioPodApi(core_v1_api=core_v1_api)

        stream = api.stream_pods(namespace="ns", label_selector="a=b", resource_version="1", timeout_seconds=1)
        await stream.__anext__()
        await stream.aclose()

        assert state["closed"] == 1


class TestReflectorFeedsReconcileLoop:
    async def test_pod_stream_drives_cell_level_reconciles(self):
        """End to end: pods from the reflector become cell reconciles, including a missed delete."""
        api = FakePodApi()
        api.list_pages.append(
            make_pod_list([make_pod("pod-0", cell="cell-a"), make_pod("pod-1", cell="cell-b")], resource_version="1")
        )
        api.stream_scripts.append(
            [raw_event("ADDED", make_pod("pod-2", cell="cell-a", resource_version="2")), FakeApiException(410)]
        )
        api.list_pages.append(make_pod_list([make_pod("pod-0", cell="cell-a")], resource_version="3"))
        api.stream_scripts.append(None)

        clock = FakeClock()
        reflector = make_reflector(api, clock=clock)
        recorder = ParentKeyRecorder()
        loop = ReconcileLoop(source=reflector.watch, reconcile=recorder, key_map=pod_cell, clock=clock)
        await loop.start()
        await settle()
        await clock.elapse(1.0)
        await settle()

        assert sorted(set(recorder.parent_keys)) == ["cell-a", "cell-b"]
        assert recorder.parent_keys.count("cell-b") >= 2
        assert [pod.metadata.name for pod in loop.get_by_parent("cell-a")] == ["pod-0"]
        assert loop.get_by_parent("cell-b") == []
        await loop.stop()

    async def test_initial_sync_barrier_holds_until_the_reflector_lists(self):
        """ReconcileLoop.start() waits for the reflector's LIST to complete."""
        api = FakePodApi()
        api.list_pages.append(RuntimeError("apiserver down"))
        api.list_pages.append(make_pod_list([make_pod("pod-0", cell="cell-a")], resource_version="1"))
        api.stream_scripts.append(None)

        clock = FakeClock()
        reflector = make_reflector(api, clock=clock, retry_delay=5.0)
        recorder = ParentKeyRecorder()
        loop = ReconcileLoop(source=reflector.watch, reconcile=recorder, key_map=pod_cell, clock=clock)
        start_task = asyncio.create_task(loop.start())
        await settle()
        assert not start_task.done()
        assert recorder.parent_keys == []

        await clock.elapse(5.0)
        await settle()
        await start_task
        assert recorder.parent_keys == ["cell-a"]
        await loop.stop()
