import logging
import time
from enum import StrEnum
from typing import NamedTuple, TypeVar
from urllib.parse import quote
from uuid import UUID

import requests
from pydantic import BaseModel, ConfigDict, ValidationError

logger = logging.getLogger(__name__)

ROUTER_REQUEST_TIMEOUT_SECONDS = 10.0
ROUTER_OPERATION_TIMEOUT_SECONDS = 240.0
ROUTER_POLL_INTERVAL_SECONDS = 1.0


class _AcceptedStatus(StrEnum):
    ACCEPTED = "accepted"


class _JobState(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    FAILED = "failed"


class RouterWorkerType(StrEnum):
    """Worker roles accepted by the SGLang router worker API."""

    REGULAR = "regular"
    PREFILL = "prefill"
    DECODE = "decode"


class RouterWorkerApi(StrEnum):
    """Worker identity formats used by supported router APIs."""

    URL = "url"
    UUID = "uuid"


class RouterWorkerJobType(StrEnum):
    """Worker lifecycle jobs reported by the router."""

    ADD = "AddWorker"
    REMOVE = "RemoveWorker"


class RouterWorkerSubmissionRejected(RuntimeError):
    """The router definitively rejected a worker registration request."""


class RouterWorkerSubmissionUnknown(RuntimeError):
    """The router may have accepted a registration whose response was lost."""


class RouterWorkerRegistrationFailed(RuntimeError):
    """The router's accepted worker registration job failed."""


class _RouterModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


_RouterModelT = TypeVar("_RouterModelT", bound=_RouterModel)


class _CreateWorkerRequest(_RouterModel):
    url: str
    worker_type: RouterWorkerType
    bootstrap_port: int | None


class _CreateWorkerResponseV02(_RouterModel):
    status: _AcceptedStatus
    worker_id: str


class _CreateWorkerResponseV03(_RouterModel):
    status: _AcceptedStatus
    worker_id: UUID
    url: str
    location: str


class _DeleteWorkerResponse(_RouterModel):
    status: _AcceptedStatus
    worker_id: str


class _JobStatus(_RouterModel):
    job_type: RouterWorkerJobType
    worker_url: str
    status: _JobState
    message: str | None


class _WorkerInfo(_RouterModel):
    id: str
    url: str
    is_healthy: bool
    job_status: _JobStatus | None = None


class _ListedWorkerV03(_RouterModel):
    id: UUID
    url: str


class _ListWorkersResponseV03(_RouterModel):
    workers: list[_ListedWorkerV03]


class RouterWorkerRegistration(NamedTuple):
    """Identity of one worker accepted by the router.

    Args:
        worker_id: Worker URL for Router 0.2.x or router-assigned UUID for
            Router 0.3.x.
        url: Canonical worker URL supplied during registration.
    """

    worker_id: str
    url: str


class _RemovalReconciliation(NamedTuple):
    is_absent: bool
    registration_observed: bool


class RouterWorkerClient:
    """Manage the asynchronous SGLang Router 0.2.2+ worker lifecycle.

    Args:
        router_url: Base HTTP URL of the router.
        worker_api: Worker identity format used by the router.
        request_timeout_seconds: Timeout for each HTTP request.
        operation_timeout_seconds: Overall deadline for one add or remove operation.
        poll_interval_seconds: Delay between worker-state observations.
    """

    def __init__(
        self,
        *,
        router_url: str,
        worker_api: RouterWorkerApi,
        request_timeout_seconds: float,
        operation_timeout_seconds: float,
        poll_interval_seconds: float,
    ) -> None:
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        if operation_timeout_seconds <= 0:
            raise ValueError("operation_timeout_seconds must be positive")
        if poll_interval_seconds < 0:
            raise ValueError("poll_interval_seconds must be nonnegative")

        self._router_url = router_url.rstrip("/")
        self._worker_api = worker_api
        self._request_timeout_seconds = request_timeout_seconds
        self._operation_timeout_seconds = operation_timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds

    def submit_registration(
        self,
        *,
        worker_url: str,
        worker_type: RouterWorkerType,
        bootstrap_port: int | None,
    ) -> RouterWorkerRegistration:
        """Submit one worker registration operation.

        Args:
            worker_url: HTTP URL of the initialized worker.
            worker_type: SGLang worker type.
            bootstrap_port: Prefill bootstrap port, or None for other worker types.

        Returns:
            The accepted router worker identity.

        Raises:
            RuntimeError: The router rejects the operation or reports invalid state.
            TimeoutError: Submission exceeds the operation deadline.
            requests.RequestException: The submission request fails.
        """
        request = _CreateWorkerRequest(
            url=worker_url,
            worker_type=worker_type,
            bootstrap_port=bootstrap_port,
        )
        deadline = time.monotonic() + self._operation_timeout_seconds

        with requests.Session() as session:
            response = session.post(
                f"{self._router_url}/workers",
                json=request.model_dump(mode="json", exclude_none=True),
                timeout=self._request_timeout(deadline=deadline, operation="register worker"),
            )
            if 400 <= response.status_code < 500 and response.status_code not in {408, 429}:
                try:
                    self._require_status(response, expected_status=202, operation="register worker")
                except (requests.HTTPError, RuntimeError) as error:
                    raise RouterWorkerSubmissionRejected("Router rejected worker registration") from error
            self._require_status(response, expected_status=202, operation="register worker")

            if self._worker_api is RouterWorkerApi.URL:
                accepted_v02 = self._parse_model(
                    model_type=_CreateWorkerResponseV02,
                    response=response,
                    operation="register worker",
                )
                if accepted_v02.worker_id != worker_url:
                    raise RuntimeError(
                        f"Router accepted worker identity {accepted_v02.worker_id!r}, expected {worker_url!r}"
                    )
                return RouterWorkerRegistration(worker_id=accepted_v02.worker_id, url=worker_url)

            accepted_v03 = self._parse_model(
                model_type=_CreateWorkerResponseV03,
                response=response,
                operation="register worker",
            )
            if accepted_v03.url != worker_url:
                raise RuntimeError(f"Router accepted worker URL {accepted_v03.url!r}, expected {worker_url!r}")

            worker_id = str(accepted_v03.worker_id)
            expected_location = f"/workers/{worker_id}"
            if accepted_v03.location != expected_location:
                raise RuntimeError(
                    f"Router returned worker location {accepted_v03.location!r}, expected {expected_location!r}"
                )

            return RouterWorkerRegistration(worker_id=worker_id, url=worker_url)

    def wait_until_active(self, *, registration: RouterWorkerRegistration) -> None:
        """Wait until an accepted worker becomes healthy and routable.

        Args:
            registration: Accepted worker identity to observe.

        Raises:
            RuntimeError: The router reports failure or a different worker.
            TimeoutError: Publication is not confirmed before the deadline.
        """
        deadline = time.monotonic() + self._operation_timeout_seconds
        with requests.Session() as session:
            self._wait_until_active(
                session=session,
                location=self._worker_location(registration=registration),
                registration=registration,
                deadline=deadline,
            )

    def find_registration_by_url(self, *, worker_url: str) -> RouterWorkerRegistration | None:
        """Wait for a worker URL to become observable in the router registry.

        This reconciles an accepted Router 0.3 registration when the POST
        response, including its UUID, was lost.

        Args:
            worker_url: Canonical worker URL supplied during registration.

        Returns:
            The observed worker identity, or None if it remains absent through
            the operation deadline.
        """
        deadline = time.monotonic() + self._operation_timeout_seconds
        with requests.Session() as session:
            while True:
                response = self._get_router_response(
                    session=session,
                    location="/workers",
                    deadline=deadline,
                    operation="list workers",
                )
                self._require_status(response, expected_status=200, operation="list workers")
                workers = self._parse_model(
                    model_type=_ListWorkersResponseV03,
                    response=response,
                    operation="list workers",
                )
                matches = [worker for worker in workers.workers if worker.url == worker_url]
                if len(matches) > 1:
                    raise RuntimeError(f"Router returned multiple workers for URL {worker_url!r}")
                if matches:
                    return RouterWorkerRegistration(worker_id=str(matches[0].id), url=worker_url)

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.warning("Router worker URL %s did not become observable during cleanup", worker_url)
                    return None
                time.sleep(min(self._poll_interval_seconds, remaining))

    def remove(
        self,
        *,
        registration: RouterWorkerRegistration,
        registration_may_be_pending: bool,
    ) -> None:
        """Remove a worker and wait until the router no longer exposes it.

        Args:
            registration: Accepted worker identity to remove.
            registration_may_be_pending: Whether registration was accepted but
                has not yet been observed as active.

        Raises:
            RuntimeError: The router rejects the operation or reports invalid state.
            TimeoutError: The router does not confirm removal before the deadline.
            requests.RequestException: The removal request fails.
        """
        location = self._worker_location(registration=registration)
        deadline = time.monotonic() + self._operation_timeout_seconds
        with requests.Session() as session:
            while True:
                try:
                    response = session.delete(
                        f"{self._router_url}{location}",
                        timeout=self._request_timeout(deadline=deadline, operation="remove worker"),
                    )
                except requests.RequestException as error:
                    logger.warning("Ambiguous router worker removal submission: %s", error)
                else:
                    if response.status_code == 404:
                        if not registration_may_be_pending:
                            return
                        logger.warning(
                            "Router worker %s is not observable while registration may still be pending",
                            registration.worker_id,
                        )
                    elif response.status_code in {408, 429} or response.status_code >= 500:
                        logger.warning(
                            "Ambiguous router worker removal response: status=%s",
                            response.status_code,
                        )
                    else:
                        self._require_status(response, expected_status=202, operation="remove worker")
                        accepted = self._parse_model(
                            model_type=_DeleteWorkerResponse,
                            response=response,
                            operation="remove worker",
                        )
                        if accepted.worker_id != registration.worker_id:
                            raise RuntimeError(
                                f"Router accepted removal for worker {accepted.worker_id}, "
                                f"expected {registration.worker_id}"
                            )
                        self._wait_until_absent(
                            session=session,
                            location=location,
                            registration=registration,
                            deadline=deadline,
                        )
                        return

                reconciliation = self._reconcile_ambiguous_removal(
                    session=session,
                    location=location,
                    registration=registration,
                    registration_may_be_pending=registration_may_be_pending,
                    deadline=deadline,
                )
                if reconciliation.is_absent:
                    return
                if reconciliation.registration_observed:
                    registration_may_be_pending = False
                self._sleep_or_timeout(deadline=deadline, operation="remove worker")

    def _wait_until_active(
        self,
        *,
        session: requests.Session,
        location: str,
        registration: RouterWorkerRegistration,
        deadline: float,
    ) -> None:
        while True:
            response = self._get_router_response(
                session=session,
                location=location,
                deadline=deadline,
                operation="observe worker",
            )
            if response.status_code == 404:
                self._sleep_or_timeout(deadline=deadline, operation="register worker")
                continue

            self._require_status(response, expected_status=200, operation="observe registered worker")
            worker = self._parse_model(
                model_type=_WorkerInfo,
                response=response,
                operation="observe registered worker",
            )
            self._validate_worker(worker=worker, registration=registration)

            if worker.job_status is not None:
                self._validate_job(
                    job_status=worker.job_status,
                    expected_job_type=RouterWorkerJobType.ADD,
                    registration=registration,
                )
                if worker.job_status.status is _JobState.FAILED:
                    detail = worker.job_status.message or "no failure message"
                    raise RouterWorkerRegistrationFailed(
                        f"Router failed to register worker {registration.worker_id}: {detail}"
                    )
            elif worker.is_healthy:
                return

            self._sleep_or_timeout(deadline=deadline, operation="register worker")

    def _wait_until_absent(
        self,
        *,
        session: requests.Session,
        location: str,
        registration: RouterWorkerRegistration,
        deadline: float,
    ) -> None:
        while True:
            response = self._get_router_response(
                session=session,
                location=location,
                deadline=deadline,
                operation="observe worker",
            )
            if response.status_code == 404:
                return

            self._require_status(response, expected_status=200, operation="observe removed worker")
            worker = self._parse_model(
                model_type=_WorkerInfo,
                response=response,
                operation="observe removed worker",
            )
            self._validate_worker(worker=worker, registration=registration)
            if worker.job_status is not None:
                if worker.job_status.job_type is RouterWorkerJobType.ADD:
                    self._validate_job(
                        job_status=worker.job_status,
                        expected_job_type=RouterWorkerJobType.ADD,
                        registration=registration,
                    )
                    self._sleep_or_timeout(deadline=deadline, operation="remove worker")
                    continue
                self._validate_job(
                    job_status=worker.job_status,
                    expected_job_type=RouterWorkerJobType.REMOVE,
                    registration=registration,
                )
                if worker.job_status.status is _JobState.FAILED:
                    detail = worker.job_status.message or "no failure message"
                    raise RuntimeError(f"Router failed to remove worker {registration.worker_id}: {detail}")

            self._sleep_or_timeout(deadline=deadline, operation="remove worker")

    def _reconcile_ambiguous_removal(
        self,
        *,
        session: requests.Session,
        location: str,
        registration: RouterWorkerRegistration,
        registration_may_be_pending: bool,
        deadline: float,
    ) -> _RemovalReconciliation:
        registration_observed = False
        while True:
            response = self._get_router_response(
                session=session,
                location=location,
                deadline=deadline,
                operation="observe worker",
            )
            if response.status_code == 404:
                return _RemovalReconciliation(
                    is_absent=registration_observed or not registration_may_be_pending,
                    registration_observed=registration_observed,
                )

            self._require_status(response, expected_status=200, operation="reconcile removed worker")
            worker = self._parse_model(
                model_type=_WorkerInfo,
                response=response,
                operation="reconcile removed worker",
            )
            self._validate_worker(worker=worker, registration=registration)
            registration_observed = True
            if worker.job_status is None:
                return _RemovalReconciliation(is_absent=False, registration_observed=True)

            if worker.job_status.job_type is RouterWorkerJobType.ADD:
                if worker.job_status.worker_url != registration.url:
                    raise RuntimeError(
                        f"Router reported a job for worker URL "
                        f"{worker.job_status.worker_url!r}, expected "
                        f"{registration.url!r}"
                    )
                if worker.job_status.status is _JobState.FAILED:
                    return _RemovalReconciliation(is_absent=False, registration_observed=True)
                self._sleep_or_timeout(deadline=deadline, operation="remove worker")
                continue

            self._validate_job(
                job_status=worker.job_status,
                expected_job_type=RouterWorkerJobType.REMOVE,
                registration=registration,
            )
            if worker.job_status.status is _JobState.FAILED:
                detail = worker.job_status.message or "no failure message"
                raise RuntimeError(f"Router failed to remove worker {registration.worker_id}: {detail}")
            self._sleep_or_timeout(deadline=deadline, operation="remove worker")

    def _get_router_response(
        self,
        *,
        session: requests.Session,
        location: str,
        deadline: float,
        operation: str,
    ) -> requests.Response:
        last_error: requests.RequestException | None = None
        while time.monotonic() < deadline:
            try:
                response = session.get(
                    f"{self._router_url}{location}",
                    timeout=self._request_timeout(deadline=deadline, operation=operation),
                )
                if response.status_code in {408, 429} or response.status_code >= 500:
                    self._raise_for_status(response=response, operation=operation)
                return response
            except requests.RequestException as error:
                last_error = error
                logger.warning("Transient router %s failure: %s", operation, error)
                self._sleep_or_timeout(deadline=deadline, operation=operation)

        raise TimeoutError(f"Timed out while attempting to {operation}") from last_error

    def _sleep_or_timeout(self, *, deadline: float, operation: str) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Timed out after {self._operation_timeout_seconds}s while attempting to {operation}")
        time.sleep(min(self._poll_interval_seconds, remaining))

    def _request_timeout(self, *, deadline: float, operation: str) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Timed out after {self._operation_timeout_seconds}s while attempting to {operation}")
        return min(self._request_timeout_seconds, remaining)

    def _worker_location(self, *, registration: RouterWorkerRegistration) -> str:
        if self._worker_api is RouterWorkerApi.URL:
            return f"/workers/{quote(registration.worker_id, safe='')}"
        return f"/workers/{registration.worker_id}"

    @staticmethod
    def _validate_worker(
        *,
        worker: _WorkerInfo,
        registration: RouterWorkerRegistration,
    ) -> None:
        if worker.id != registration.worker_id:
            raise RuntimeError(f"Router returned worker {worker.id}, expected {registration.worker_id}")
        if worker.url != registration.url:
            raise RuntimeError(f"Router returned worker URL {worker.url!r}, expected {registration.url!r}")

    @staticmethod
    def _validate_job(
        *,
        job_status: _JobStatus,
        expected_job_type: RouterWorkerJobType,
        registration: RouterWorkerRegistration,
    ) -> None:
        if job_status.job_type is not expected_job_type:
            raise RuntimeError(f"Router reported {job_status.job_type} while waiting for {expected_job_type}")
        if job_status.worker_url != registration.url:
            raise RuntimeError(
                f"Router reported a job for worker URL {job_status.worker_url!r}, " f"expected {registration.url!r}"
            )

    @staticmethod
    def _parse_model(
        *,
        model_type: type[_RouterModelT],
        response: requests.Response,
        operation: str,
    ) -> _RouterModelT:
        try:
            return model_type.model_validate_json(response.text)
        except ValidationError as error:
            raise RuntimeError(
                f"Router returned an invalid response while attempting to {operation}: {response.text}"
            ) from error

    @classmethod
    def _require_status(
        cls,
        response: requests.Response,
        *,
        expected_status: int,
        operation: str,
    ) -> None:
        if response.status_code == expected_status:
            return
        cls._raise_for_status(response=response, operation=operation)
        raise RuntimeError(
            f"Router returned HTTP {response.status_code} while attempting to {operation}; "
            f"expected HTTP {expected_status}: {response.text}"
        )

    @staticmethod
    def _raise_for_status(*, response: requests.Response, operation: str) -> None:
        try:
            response.raise_for_status()
        except requests.HTTPError as error:
            error.add_note(f"Router response while attempting to {operation}: {response.text}")
            raise
