from collections.abc import Sequence

from tests.utils.soak.state import (
    Event,
    SoakActionAppliedEvent,
    SoakActionRequest,
    SoakActionRequestedEvent,
    SoakActionResultEvent,
)


def expand_fault_batches(events: Sequence[Event]) -> list[Event]:
    batches: dict[str, list[SoakActionRequest]] = {}
    expanded: list[Event] = []
    for event in events:
        if isinstance(event, SoakActionRequestedEvent):
            additional = event.request.additional_requests
            if additional:
                validate_fault_batch(event.request)
                batches[event.request.request_id] = additional
                event = event.model_copy(
                    update={"request": event.request.model_copy(update={"additional_requests": []})}
                )
            expanded.append(event)
            expanded.extend(
                SoakActionRequestedEvent(timestamp=event.timestamp, request=request) for request in additional
            )
        elif isinstance(event, SoakActionAppliedEvent) and event.request_id in batches:
            receipts = event.evidence["batch_receipts"]
            requests = batches[event.request_id]
            if set(receipts) != {request.request_id for request in requests}:
                raise ValueError("Batch effect receipts do not cover exactly its additional requests")
            expanded.append(
                event.model_copy(
                    update={
                        "evidence": {key: value for key, value in event.evidence.items() if key != "batch_receipts"}
                    }
                )
            )
            expanded.extend(
                SoakActionAppliedEvent(
                    timestamp=event.timestamp, request_id=request.request_id, evidence=receipts[request.request_id]
                )
                for request in requests
            )
        elif isinstance(event, SoakActionResultEvent) and event.request_id in batches:
            expanded.append(event)
            expanded.extend(
                event.model_copy(update={"request_id": request.request_id}) for request in batches[event.request_id]
            )
        else:
            expanded.append(event)
    return expanded


def validate_fault_batch(request: SoakActionRequest) -> None:
    requests = [request, *request.additional_requests]
    if any(child.additional_requests for child in request.additional_requests):
        raise ValueError("Fault batches cannot be nested")
    if len({child.request_id for child in requests}) != len(requests):
        raise ValueError("Fault batch request identities must be distinct")
    if any(not isinstance(child.target, dict) or not child.harms_cell for child in requests):
        raise ValueError("Fault batches require harmful cell requests")
    names = [child.target["metadata"]["name"] for child in requests]
    if len(set(names)) != len(names):
        raise ValueError("Fault batches require distinct cells")
