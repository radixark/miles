from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from miles.utils.workers.k8s_types import ContainerStatus, Pod, PodMetadata, PodStatus
from miles.utils.workers.reconcile.k8s_api import PodWatchEvent, exception_rejects_cursor


STARTED_AT = datetime(2026, 1, 1, 12, 0, 0)


def make_exception(**fields: Any) -> Exception:
    exception = Exception("boom")
    for name, value in fields.items():
        setattr(exception, name, value)
    return exception


def make_wire_pod() -> SimpleNamespace:
    return SimpleNamespace(
        metadata=SimpleNamespace(name="pod-0", uid="uid-0", resource_version="7", labels=None, annotations=None),
        spec=None,
        status=SimpleNamespace(
            pod_ip="10.0.0.1", conditions=None, container_statuses=[SimpleNamespace(restart_count=2)]
        ),
    )


class TestResourceVersionParsing:
    @pytest.mark.parametrize(
        ("obj", "expected"),
        [
            (SimpleNamespace(metadata=SimpleNamespace(resource_version="7")), "7"),
            (dict(metadata=dict(resourceVersion="7")), "7"),
        ],
    )
    def test_both_wire_shapes_are_read(self, obj: Any, expected: str) -> None:
        """A deserialized model spells it as an attribute, a raw dict as a camelCase key."""
        assert PodWatchEvent.from_frame(event_type="BOOKMARK", obj=obj).resource_version == expected

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
    def test_a_frame_without_a_readable_version_parses_to_none(self, obj: Any) -> None:
        """A missing or malformed metadata block must parse to None, never raise: the caller keeps its cursor."""
        assert PodWatchEvent.from_frame(event_type="BOOKMARK", obj=obj).resource_version is None

    def test_a_payload_that_is_not_an_object_at_all_ends_the_stream(self) -> None:
        """A frame no parser can read is not a frame with a missing field, and the watch must not go on."""
        with pytest.raises(ValidationError):
            PodWatchEvent.from_frame(event_type="BOOKMARK", obj="a payload that is not an object at all")


class TestPodParsing:
    def test_a_pod_frame_carries_the_pod_the_wire_described(self) -> None:
        """Every consumer downstream reads typed fields, so the frame is validated once, here."""
        event = PodWatchEvent.from_frame(event_type="ADDED", obj=make_wire_pod())

        assert event.pod == Pod(
            metadata=PodMetadata(name="pod-0", uid="uid-0", resource_version="7"),
            status=PodStatus(pod_ip="10.0.0.1", container_statuses=[ContainerStatus(restart_count=2)]),
        )

    def test_a_raw_json_pod_is_read_through_its_camel_case_spelling(self) -> None:
        """A payload the client left as raw JSON spells every compound field differently."""
        obj = dict(
            metadata=dict(name="pod-0", uid="uid-0", resourceVersion="7"),
            spec=dict(nodeName="gpu-1"),
            status=dict(podIP="10.0.0.1", containerStatuses=[dict(restartCount=2)]),
        )

        event = PodWatchEvent.from_frame(event_type="ADDED", obj=obj)

        assert event.pod is not None
        assert (event.pod.spec.node_name, event.pod.status.pod_ip) == ("gpu-1", "10.0.0.1")
        assert [status.restart_count for status in event.pod.status.container_statuses] == [2]

    def test_the_scheduling_fields_a_gated_pod_is_paired_by_are_typed(self) -> None:
        """Colocate pairing reads the gates and the node selector off the pod, so PodSpec has to model them."""
        obj = dict(
            metadata=dict(name="pod-0", uid="uid-0"),
            spec=dict(
                nodeName=None,
                schedulingGates=[dict(name="miles.ai/awaiting-pair")],
                nodeSelector={"kubernetes.io/hostname": "gpu-1"},
            ),
        )

        event = PodWatchEvent.from_frame(event_type="ADDED", obj=obj)

        assert event.pod is not None
        assert [gate.name for gate in event.pod.spec.scheduling_gates] == ["miles.ai/awaiting-pair"]
        assert event.pod.spec.node_selector == {"kubernetes.io/hostname": "gpu-1"}

    def test_an_ungated_pod_reads_as_empty_rather_than_missing(self) -> None:
        """The client leaves both fields out entirely, and pairing must not have to guard every read."""
        event = PodWatchEvent.from_frame(event_type="ADDED", obj=make_wire_pod())

        assert event.pod is not None
        assert (event.pod.spec.scheduling_gates, event.pod.spec.node_selector) == ([], {})

    def test_a_running_container_the_client_deserialized_is_read_as_running(self) -> None:
        """The client hands the running block back as a nested object rather than a mapping, and a
        state miles cannot parse poisons every frame from the moment a container starts, which is
        every frame the reconcile loop is watching for."""
        obj = make_wire_pod()
        obj.status.container_statuses = [
            SimpleNamespace(
                restart_count=0, state=SimpleNamespace(running=SimpleNamespace(started_at=STARTED_AT), terminated=None)
            )
        ]

        event = PodWatchEvent.from_frame(event_type="ADDED", obj=obj)

        assert event.pod is not None
        running = event.pod.status.container_statuses[0].state.running
        assert running is not None and running.started_at == STARTED_AT

    def test_a_running_container_left_as_raw_json_is_read_the_same_way(self) -> None:
        """The two wire shapes spell the start time differently and both have to reach the model."""
        obj = dict(
            metadata=dict(name="pod-0", uid="uid-0"),
            status=dict(containerStatuses=[dict(restartCount=0, state=dict(running=dict(startedAt="2026-01-01")))]),
        )

        event = PodWatchEvent.from_frame(event_type="ADDED", obj=obj)

        assert event.pod is not None
        running = event.pod.status.container_statuses[0].state.running
        assert running is not None and running.started_at == datetime(2026, 1, 1)

    @pytest.mark.parametrize("event_type", ["BOOKMARK", "ERROR"])
    def test_a_frame_that_is_not_about_a_pod_carries_no_pod(self, event_type: str) -> None:
        """BOOKMARK and ERROR frames carry a bare version and a Status, and parsing them as pods would fail."""
        assert PodWatchEvent.from_frame(event_type=event_type, obj=dict(code=410, reason="Expired")).pod is None

    def test_a_pod_frame_miles_cannot_read_preserves_its_cursor_and_error(self) -> None:
        """A malformed pod payload keeps its readable envelope so the reflector can skip past it explicitly."""
        obj = make_wire_pod()
        obj.status.container_statuses = [SimpleNamespace(name="main", state=None)]

        event = PodWatchEvent.from_frame(event_type="ADDED", obj=obj)

        assert event.pod is None
        assert event.resource_version == "7"
        assert event.pod_validation_error is not None
        assert "restart_count" in event.pod_validation_error


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
            ("BOOKMARK", dict(code=410)),
        ],
    )
    def test_anything_else_leaves_the_cursor_alone(self, event_type: str, obj: Any) -> None:
        """Only an ERROR frame may invalidate a cursor, and only for a cursor-specific code."""
        assert not PodWatchEvent.from_frame(event_type=event_type, obj=obj).rejects_cursor

    def test_a_pod_frame_carrying_a_dead_cursor_code_leaves_the_cursor_alone(self) -> None:
        """A pod whose own fields happen to spell 410 must not be read as an expired-cursor error."""
        obj = make_wire_pod()
        obj.code = 410
        obj.reason = "Expired"

        event = PodWatchEvent.from_frame(event_type="MODIFIED", obj=obj)

        assert not event.rejects_cursor
        assert event.pod is not None and event.pod.metadata.name == "pod-0"


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
