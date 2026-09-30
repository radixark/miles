import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import get_args

import pytest

from miles.utils.test_utils.fault_injector.actions.union import FaultAction
from miles.utils.test_utils.fault_injector.models import FaultHookName, FaultHookRequest
from miles.utils.test_utils.fault_injector.static_source import read_declared_fault_hooks
from miles.utils.workers.naming import compute_cell_id, parse_cell_id

REPO_ROOT = Path(__file__).resolve().parents[3]
DEBUG_DOC = REPO_ROOT / "docs" / "developer" / "debug.md"
_PARAGRAPH_MARKER = "**Fault injection.**"
_SUPPORTED_ACTIONS = {model.model_fields["kind"].default for model in get_args(get_args(FaultAction)[0])}


@pytest.fixture(scope="module")
def fault_injection_paragraph() -> str:
    paragraphs = [p for p in DEBUG_DOC.read_text().split("\n\n") if p.startswith(_PARAGRAPH_MARKER)]
    assert len(paragraphs) == 1
    return paragraphs[0]


@pytest.fixture(scope="module")
def documented_requests(fault_injection_paragraph: str) -> list[dict[str, object]]:
    examples = re.findall(r"`(\[.*?\])`", fault_injection_paragraph, flags=re.DOTALL)
    assert examples, "The fault injection paragraph must show a JSON request example"
    return [request for example in examples for request in json.loads(example)]


class TestDocumentedFaultInjectionExample:
    def test_the_documented_example_loads_through_the_real_request_parser(
        self, documented_requests: list[dict[str, object]]
    ) -> None:
        """The documented command must load through the same parser used by workers."""
        requests = read_declared_fault_hooks(
            SimpleNamespace(ci_fault_hooks=json.dumps(documented_requests), ci_fault_hooks_path=None)
        )

        assert len(requests) == len(documented_requests)
        assert all(isinstance(request, FaultHookRequest) for request in requests)

    def test_the_documented_example_targets_a_cell_by_id(self, documented_requests: list[dict[str, object]]) -> None:
        """The example must identify the actual cell instead of a retired index field."""
        for request in documented_requests:
            cell_id = request["action"]["cell_id"]
            parsed = parse_cell_id(cell_id)
            assert compute_cell_id(pool_id=parsed.pool_id, cell_index=parsed.cell_index) == cell_id
            assert parsed.cell_index >= 0

    def test_the_paragraph_documents_exactly_the_supported_actions(self, fault_injection_paragraph: str) -> None:
        """Every action kind must remain discoverable in the fault injection paragraph."""
        identifiers = {
            span for span in re.findall(r"`([^`]+)`", fault_injection_paragraph) if re.fullmatch(r"[a-z][a-z_]*", span)
        }

        assert identifiers - {hook.value for hook in FaultHookName} == _SUPPORTED_ACTIONS

    def test_the_paragraph_never_mentions_the_retired_cell_index_field(self, fault_injection_paragraph: str) -> None:
        """The prose must not advertise a field rejected by the strict request parser."""
        assert "cell_index" not in fault_injection_paragraph
