"""The output store refs one session saw, and which of them its client took."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import orjson

from miles.utils.output_store.dropped import store_refs_of
from miles.utils.output_store.plan import OUTPUT_STORE_REF_KEY, OUTPUT_STORE_REQUEST_KEY, store_ref_from_dict

if TYPE_CHECKING:
    from radix_transfer_plugins import StoreRef


@dataclass
class SessionStoreRefs:
    """Every ref the session's responses carried, and the keys its ``/samples`` replies handed over.

    A successful ``/samples`` reply hands the refs of its samples to the client, and no later reply
    takes them back. The session's DELETE deletes the rest (aborted turns, rolled-back records, leaves
    the picker dropped), or every ref when the client reports it received no reply; refs arriving after
    DELETE began are deleted at once.
    """

    arrived: dict[str, StoreRef] = field(default_factory=dict)
    handed_over: set[str] = field(default_factory=set)

    def hand_over(self, samples: Iterable[Any]) -> None:
        self.handed_over.update(store_refs_of(samples))

    def kept_by_session(self, *, samples_received: bool) -> list[StoreRef]:
        if not samples_received:
            return list(self.arrived.values())
        return [ref for key, ref in self.arrived.items() if key not in self.handed_over]


def response_store_ref(request_body: dict, result: dict) -> StoreRef | None:
    """The ref of a proxied chat response that asked for the output store; aborts carry one too."""
    if not request_body.get(OUTPUT_STORE_REQUEST_KEY) or result["status_code"] != 200:
        return None
    try:
        choice = orjson.loads(result["response_body"])["choices"][0]
    except (orjson.JSONDecodeError, KeyError, IndexError, TypeError):
        return None  # extract_completion rejects the response; its ref, if any, is lost
    return store_ref_from_dict((choice.get("meta_info") or {}).get(OUTPUT_STORE_REF_KEY))
