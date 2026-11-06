import json
import time
from collections.abc import Iterator
from unittest.mock import MagicMock
from urllib.parse import quote

import pytest
import requests
from tests.ci.ci_register import register_cpu_ci

from miles.backends.sglang_utils.router_worker_client import (
    RouterWorkerApi,
    RouterWorkerClient,
    RouterWorkerJobType,
    RouterWorkerRegistration,
    RouterWorkerSubmissionRejected,
    RouterWorkerType,
)

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

_WORKER_ID = "12345678-1234-5678-1234-567812345678"
_WORKER_URL = "http://127.0.0.1:30000"
_LOCATION = f"/workers/{_WORKER_ID}"


class _Response:
    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self.text = json.dumps(payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"HTTP {self.status_code}",
                response=self,
            )


class _Session:
    def __init__(
        self,
        *,
        post_responses: list[_Response] | None = None,
        get_responses: list[_Response] | None = None,
        delete_responses: list[_Response | requests.RequestException] | None = None,
    ) -> None:
        self._post_responses: Iterator[_Response] = iter(post_responses or [])
        self._get_responses: Iterator[_Response] = iter(get_responses or [])
        self._delete_responses: Iterator[_Response | requests.RequestException] = iter(delete_responses or [])
        self.calls: list[tuple[str, str, object | None, float]] = []

    def __enter__(self) -> "_Session":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        return None

    def post(self, url: str, *, json: object, timeout: float) -> _Response:
        self.calls.append(("POST", url, json, timeout))
        return next(self._post_responses)

    def get(self, url: str, *, timeout: float) -> _Response:
        self.calls.append(("GET", url, None, timeout))
        return next(self._get_responses)

    def delete(self, url: str, *, timeout: float) -> _Response:
        self.calls.append(("DELETE", url, None, timeout))
        response = next(self._delete_responses)
        if isinstance(response, requests.RequestException):
            raise response
        return response


def _client(
    *,
    worker_api: RouterWorkerApi = RouterWorkerApi.UUID,
    poll_interval_seconds: float = 0.0,
) -> RouterWorkerClient:
    return RouterWorkerClient(
        router_url="http://router:3000",
        worker_api=worker_api,
        request_timeout_seconds=2.0,
        operation_timeout_seconds=10.0,
        poll_interval_seconds=poll_interval_seconds,
    )


def _create_response() -> _Response:
    return _Response(
        202,
        {
            "status": "accepted",
            "worker_id": str(_WORKER_ID),
            "url": _WORKER_URL,
            "location": _LOCATION,
            "message": "Worker addition queued for background processing",
        },
    )


def _worker_response(
    *,
    is_healthy: bool,
    job_status: dict[str, object] | None,
    job_type: RouterWorkerJobType = RouterWorkerJobType.ADD,
) -> _Response:
    if job_status is not None:
        job_status = {
            "job_type": job_type,
            "worker_url": _WORKER_URL,
            **job_status,
        }
    return _Response(
        200,
        {
            "id": str(_WORKER_ID),
            "url": _WORKER_URL,
            "is_healthy": is_healthy,
            "job_status": job_status,
        },
    )


def test_register_waits_for_confirmed_active_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _Session(
        post_responses=[_create_response()],
        get_responses=[
            _worker_response(
                is_healthy=False,
                job_status={"status": "pending", "message": None},
            ),
            _worker_response(
                is_healthy=False,
                job_status={"status": "processing", "message": None},
            ),
            _worker_response(is_healthy=True, job_status=None),
        ],
    )
    monkeypatch.setattr(requests, "Session", lambda: session)

    client = _client()
    result = client.submit_registration(
        worker_url=_WORKER_URL,
        worker_type=RouterWorkerType.REGULAR,
        bootstrap_port=None,
    )
    client.wait_until_active(registration=result)

    assert result == RouterWorkerRegistration(worker_id=_WORKER_ID, url=_WORKER_URL)
    assert session.calls == [
        (
            "POST",
            "http://router:3000/workers",
            {"url": _WORKER_URL, "worker_type": "regular"},
            2.0,
        ),
        ("GET", f"http://router:3000{_LOCATION}", None, 2.0),
        ("GET", f"http://router:3000{_LOCATION}", None, 2.0),
        ("GET", f"http://router:3000{_LOCATION}", None, 2.0),
    ]


def test_register_reports_background_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _Session(
        post_responses=[_create_response()],
        get_responses=[
            _worker_response(
                is_healthy=False,
                job_status={"status": "failed", "message": "Worker already exists"},
            )
        ],
    )
    monkeypatch.setattr(requests, "Session", lambda: session)

    with pytest.raises(RuntimeError, match="Worker already exists"):
        client = _client()
        registration = client.submit_registration(
            worker_url=_WORKER_URL,
            worker_type=RouterWorkerType.REGULAR,
            bootstrap_port=None,
        )
        client.wait_until_active(registration=registration)


def test_register_rejects_colliding_removal_job(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _Session(
        post_responses=[_create_response()],
        get_responses=[
            _worker_response(
                is_healthy=False,
                job_status={"status": "processing", "message": None},
                job_type=RouterWorkerJobType.REMOVE,
            )
        ],
    )
    monkeypatch.setattr(requests, "Session", lambda: session)

    client = _client()
    registration = client.submit_registration(
        worker_url=_WORKER_URL,
        worker_type=RouterWorkerType.REGULAR,
        bootstrap_port=None,
    )
    with pytest.raises(RuntimeError, match="RemoveWorker.*AddWorker"):
        client.wait_until_active(registration=registration)


def test_register_rejects_malformed_acceptance(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _Session(
        post_responses=[
            _Response(
                202,
                {
                    "status": "accepted",
                    "worker_id": "not-a-uuid",
                    "url": _WORKER_URL,
                    "location": "/workers/not-a-uuid",
                },
            )
        ]
    )
    monkeypatch.setattr(requests, "Session", lambda: session)

    with pytest.raises(RuntimeError, match="invalid response"):
        _client().submit_registration(
            worker_url=_WORKER_URL,
            worker_type=RouterWorkerType.REGULAR,
            bootstrap_port=None,
        )


def test_register_marks_client_rejection_as_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _Session(
        post_responses=[_Response(400, {"error": "invalid worker"})],
    )
    monkeypatch.setattr(requests, "Session", lambda: session)

    with pytest.raises(RouterWorkerSubmissionRejected):
        _client().submit_registration(
            worker_url=_WORKER_URL,
            worker_type=RouterWorkerType.REGULAR,
            bootstrap_port=None,
        )


def test_register_preserves_ambiguous_server_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _Session(
        post_responses=[_Response(503, {"error": "temporarily unavailable"})],
    )
    monkeypatch.setattr(requests, "Session", lambda: session)

    with pytest.raises(requests.HTTPError):
        _client().submit_registration(
            worker_url=_WORKER_URL,
            worker_type=RouterWorkerType.REGULAR,
            bootstrap_port=None,
        )


def test_register_reconciles_transient_not_found_without_second_post(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _Session(
        post_responses=[_create_response()],
        get_responses=[
            _Response(404, {"error": "not found"}),
            _worker_response(is_healthy=True, job_status=None),
        ],
    )
    monkeypatch.setattr(requests, "Session", lambda: session)

    client = _client()
    registration = client.submit_registration(
        worker_url=_WORKER_URL,
        worker_type=RouterWorkerType.REGULAR,
        bootstrap_port=None,
    )
    client.wait_until_active(registration=registration)

    assert [call[0] for call in session.calls] == ["POST", "GET", "GET"]


@pytest.mark.parametrize("status_code", [408, 429, 503])
def test_register_retries_transient_observation_error(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    session = _Session(
        post_responses=[_create_response()],
        get_responses=[
            _Response(status_code, {"error": "temporarily unavailable"}),
            _worker_response(is_healthy=True, job_status=None),
        ],
    )
    monkeypatch.setattr(requests, "Session", lambda: session)

    client = _client()
    registration = client.submit_registration(
        worker_url=_WORKER_URL,
        worker_type=RouterWorkerType.REGULAR,
        bootstrap_port=None,
    )
    client.wait_until_active(registration=registration)

    assert [call[0] for call in session.calls] == ["POST", "GET", "GET"]


def test_register_includes_prefill_bootstrap_port(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _Session(
        post_responses=[_create_response()],
        get_responses=[_worker_response(is_healthy=True, job_status=None)],
    )
    monkeypatch.setattr(requests, "Session", lambda: session)

    client = _client()
    registration = client.submit_registration(
        worker_url=_WORKER_URL,
        worker_type=RouterWorkerType.PREFILL,
        bootstrap_port=40000,
    )
    client.wait_until_active(registration=registration)

    assert session.calls[0] == (
        "POST",
        "http://router:3000/workers",
        {
            "url": _WORKER_URL,
            "worker_type": "prefill",
            "bootstrap_port": 40000,
        },
        2.0,
    )


def test_find_registration_by_url_waits_until_worker_is_observable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _Session(
        get_responses=[
            _Response(200, {"workers": [], "total": 0}),
            _Response(
                200,
                {
                    "workers": [
                        {
                            "id": _WORKER_ID,
                            "url": _WORKER_URL,
                            "is_healthy": False,
                            "job_status": None,
                        }
                    ],
                    "total": 1,
                },
            ),
        ],
    )
    monkeypatch.setattr(requests, "Session", lambda: session)

    registration = _client().find_registration_by_url(worker_url=_WORKER_URL)

    assert registration == RouterWorkerRegistration(worker_id=_WORKER_ID, url=_WORKER_URL)
    assert session.calls == [
        ("GET", "http://router:3000/workers", None, 2.0),
        ("GET", "http://router:3000/workers", None, 2.0),
    ]


def test_remove_waits_until_worker_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _Session(
        delete_responses=[
            _Response(
                202,
                {
                    "status": "accepted",
                    "worker_id": str(_WORKER_ID),
                    "message": "Worker removal queued for background processing",
                },
            )
        ],
        get_responses=[
            _worker_response(
                is_healthy=False,
                job_status={"status": "processing", "message": None},
                job_type=RouterWorkerJobType.REMOVE,
            ),
            _Response(404, {"error": "not found"}),
        ],
    )
    monkeypatch.setattr(requests, "Session", lambda: session)

    _client().remove(
        registration=RouterWorkerRegistration(worker_id=_WORKER_ID, url=_WORKER_URL),
        registration_may_be_pending=False,
    )

    assert session.calls == [
        ("DELETE", f"http://router:3000{_LOCATION}", None, 2.0),
        ("GET", f"http://router:3000{_LOCATION}", None, 2.0),
        ("GET", f"http://router:3000{_LOCATION}", None, 2.0),
    ]


def test_remove_waits_for_registration_job_to_yield_to_removal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _Session(
        delete_responses=[
            _Response(
                202,
                {
                    "status": "accepted",
                    "worker_id": str(_WORKER_ID),
                },
            )
        ],
        get_responses=[
            _worker_response(
                is_healthy=False,
                job_status={"status": "processing", "message": None},
                job_type=RouterWorkerJobType.ADD,
            ),
            _worker_response(
                is_healthy=False,
                job_status={"status": "processing", "message": None},
                job_type=RouterWorkerJobType.REMOVE,
            ),
            _Response(404, {"error": "not found"}),
        ],
    )
    monkeypatch.setattr(requests, "Session", lambda: session)

    _client().remove(
        registration=RouterWorkerRegistration(worker_id=_WORKER_ID, url=_WORKER_URL),
        registration_may_be_pending=True,
    )

    assert session.calls == [
        ("DELETE", f"http://router:3000{_LOCATION}", None, 2.0),
        ("GET", f"http://router:3000{_LOCATION}", None, 2.0),
        ("GET", f"http://router:3000{_LOCATION}", None, 2.0),
        ("GET", f"http://router:3000{_LOCATION}", None, 2.0),
    ]


def test_remove_retries_not_found_while_registration_is_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _Session(
        delete_responses=[
            _Response(404, {"error": "not found"}),
            _Response(
                202,
                {
                    "status": "accepted",
                    "worker_id": str(_WORKER_ID),
                },
            ),
        ],
        get_responses=[
            _Response(404, {"error": "not found"}),
            _worker_response(
                is_healthy=False,
                job_status={"status": "processing", "message": None},
                job_type=RouterWorkerJobType.ADD,
            ),
            _worker_response(
                is_healthy=False,
                job_status={"status": "processing", "message": None},
                job_type=RouterWorkerJobType.REMOVE,
            ),
            _Response(404, {"error": "not found"}),
        ],
    )
    monkeypatch.setattr(requests, "Session", lambda: session)

    _client().remove(
        registration=RouterWorkerRegistration(worker_id=_WORKER_ID, url=_WORKER_URL),
        registration_may_be_pending=True,
    )

    assert session.calls == [
        ("DELETE", f"http://router:3000{_LOCATION}", None, 2.0),
        ("GET", f"http://router:3000{_LOCATION}", None, 2.0),
        ("DELETE", f"http://router:3000{_LOCATION}", None, 2.0),
        ("GET", f"http://router:3000{_LOCATION}", None, 2.0),
        ("GET", f"http://router:3000{_LOCATION}", None, 2.0),
        ("GET", f"http://router:3000{_LOCATION}", None, 2.0),
    ]


def test_remove_accepts_absence_after_observing_pending_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _Session(
        delete_responses=[_Response(404, {"error": "not found"})],
        get_responses=[
            _worker_response(
                is_healthy=False,
                job_status={"status": "processing", "message": None},
                job_type=RouterWorkerJobType.ADD,
            ),
            _Response(404, {"error": "not found"}),
        ],
    )
    monkeypatch.setattr(requests, "Session", lambda: session)

    _client().remove(
        registration=RouterWorkerRegistration(worker_id=_WORKER_ID, url=_WORKER_URL),
        registration_may_be_pending=True,
    )

    assert session.calls == [
        ("DELETE", f"http://router:3000{_LOCATION}", None, 2.0),
        ("GET", f"http://router:3000{_LOCATION}", None, 2.0),
        ("GET", f"http://router:3000{_LOCATION}", None, 2.0),
    ]


def test_remove_retries_delete_for_failed_registration_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _Session(
        delete_responses=[
            requests.Timeout("response lost"),
            _Response(
                202,
                {
                    "status": "accepted",
                    "worker_id": str(_WORKER_ID),
                },
            ),
        ],
        get_responses=[
            _worker_response(
                is_healthy=False,
                job_status={"status": "failed", "message": "registration failed"},
                job_type=RouterWorkerJobType.ADD,
            ),
            _Response(404, {"error": "not found"}),
        ],
    )
    monkeypatch.setattr(requests, "Session", lambda: session)

    _client().remove(
        registration=RouterWorkerRegistration(worker_id=_WORKER_ID, url=_WORKER_URL),
        registration_may_be_pending=True,
    )

    assert session.calls == [
        ("DELETE", f"http://router:3000{_LOCATION}", None, 2.0),
        ("GET", f"http://router:3000{_LOCATION}", None, 2.0),
        ("DELETE", f"http://router:3000{_LOCATION}", None, 2.0),
        ("GET", f"http://router:3000{_LOCATION}", None, 2.0),
    ]


def test_remove_reconciles_ambiguous_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _Session(
        delete_responses=[requests.Timeout("response lost")],
        get_responses=[
            _worker_response(
                is_healthy=False,
                job_status={"status": "processing", "message": None},
                job_type=RouterWorkerJobType.REMOVE,
            ),
            _Response(404, {"error": "not found"}),
        ],
    )
    monkeypatch.setattr(requests, "Session", lambda: session)

    _client().remove(
        registration=RouterWorkerRegistration(
            worker_id=_WORKER_ID,
            url=_WORKER_URL,
        ),
        registration_may_be_pending=False,
    )

    assert session.calls == [
        ("DELETE", f"http://router:3000{_LOCATION}", None, 2.0),
        ("GET", f"http://router:3000{_LOCATION}", None, 2.0),
        ("GET", f"http://router:3000{_LOCATION}", None, 2.0),
    ]


def test_remove_throttles_retry_after_transient_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _Session(
        delete_responses=[
            _Response(503, {"error": "temporarily unavailable"}),
            _Response(
                202,
                {
                    "status": "accepted",
                    "worker_id": str(_WORKER_ID),
                },
            ),
        ],
        get_responses=[
            _worker_response(is_healthy=True, job_status=None),
            _Response(404, {"error": "not found"}),
        ],
    )
    monkeypatch.setattr(requests, "Session", lambda: session)
    sleep = MagicMock()
    monkeypatch.setattr(time, "sleep", sleep)

    _client(poll_interval_seconds=1.0).remove(
        registration=RouterWorkerRegistration(worker_id=_WORKER_ID, url=_WORKER_URL),
        registration_may_be_pending=False,
    )

    assert session.calls == [
        ("DELETE", f"http://router:3000{_LOCATION}", None, 2.0),
        ("GET", f"http://router:3000{_LOCATION}", None, 2.0),
        ("DELETE", f"http://router:3000{_LOCATION}", None, 2.0),
        ("GET", f"http://router:3000{_LOCATION}", None, 2.0),
    ]
    sleep.assert_called_once_with(1.0)


def test_remove_reports_background_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _Session(
        delete_responses=[
            _Response(
                202,
                {
                    "status": "accepted",
                    "worker_id": str(_WORKER_ID),
                },
            )
        ],
        get_responses=[
            _worker_response(
                is_healthy=False,
                job_status={"status": "failed", "message": "remove failed"},
                job_type=RouterWorkerJobType.REMOVE,
            )
        ],
    )
    monkeypatch.setattr(requests, "Session", lambda: session)

    with pytest.raises(RuntimeError, match="remove failed"):
        _client().remove(
            registration=RouterWorkerRegistration(worker_id=_WORKER_ID, url=_WORKER_URL),
            registration_may_be_pending=False,
        )


def test_remove_is_idempotent_when_worker_is_already_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _Session(delete_responses=[_Response(404, {"error": "not found"})])
    monkeypatch.setattr(requests, "Session", lambda: session)

    _client().remove(
        registration=RouterWorkerRegistration(worker_id=_WORKER_ID, url=_WORKER_URL),
        registration_may_be_pending=False,
    )

    assert session.calls == [
        ("DELETE", f"http://router:3000{_LOCATION}", None, 2.0),
    ]


def test_remove_reports_definitive_client_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _Session(delete_responses=[_Response(400, {"error": "invalid worker"})])
    monkeypatch.setattr(requests, "Session", lambda: session)

    with pytest.raises(requests.HTTPError, match="400"):
        _client().remove(
            registration=RouterWorkerRegistration(worker_id=_WORKER_ID, url=_WORKER_URL),
            registration_may_be_pending=False,
        )

    assert session.calls == [
        ("DELETE", f"http://router:3000{_LOCATION}", None, 2.0),
    ]


def test_router_0_2_uses_encoded_worker_url_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded_worker_url = quote(_WORKER_URL, safe="")
    location = f"/workers/{encoded_worker_url}"
    session = _Session(
        post_responses=[
            _Response(
                202,
                {
                    "status": "accepted",
                    "worker_id": _WORKER_URL,
                },
            )
        ],
        get_responses=[
            _Response(
                200,
                {
                    "id": _WORKER_URL,
                    "url": _WORKER_URL,
                    "is_healthy": True,
                    "job_status": None,
                },
            ),
            _Response(404, {"error": "not found"}),
        ],
        delete_responses=[
            _Response(
                202,
                {
                    "status": "accepted",
                    "worker_id": _WORKER_URL,
                },
            )
        ],
    )
    monkeypatch.setattr(requests, "Session", lambda: session)

    client = _client(worker_api=RouterWorkerApi.URL)
    registration = client.submit_registration(
        worker_url=_WORKER_URL,
        worker_type=RouterWorkerType.REGULAR,
        bootstrap_port=None,
    )
    client.wait_until_active(registration=registration)
    client.remove(registration=registration, registration_may_be_pending=False)

    assert registration == RouterWorkerRegistration(
        worker_id=_WORKER_URL,
        url=_WORKER_URL,
    )
    assert session.calls == [
        (
            "POST",
            "http://router:3000/workers",
            {"url": _WORKER_URL, "worker_type": "regular"},
            2.0,
        ),
        ("GET", f"http://router:3000{location}", None, 2.0),
        ("DELETE", f"http://router:3000{location}", None, 2.0),
        ("GET", f"http://router:3000{location}", None, 2.0),
    ]
