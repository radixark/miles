"""Test helpers for the ft controller test suite.

Submodules:
- controller_fakes: platform fakes, test detectors, controller harness
- diagnostic_fakes: diagnostic stubs, scheduler fakes, node agent fakes
- metric_injectors: metric store/wandb factories, inject_* helpers
- agent_fakes: collector fakes, HW-collector mocks, stack trace helpers
"""
from tests.fast.utils.ft.helpers.agent_fakes import (
    SAMPLE_PYSPY_OUTPUT_DIFFERENT_STUCK,
    SAMPLE_PYSPY_OUTPUT_NORMAL,
    SAMPLE_PYSPY_OUTPUT_STUCK,
    FakeKmsgReader,
    TestCollector,
    create_sysfs_interface,
    make_mock_pynvml,
    make_rank_pids_provider,
    make_trace_result,
)
from tests.fast.utils.ft.helpers.controller_fakes import (
    AlwaysMarkBadDetector,
    AlwaysNoneDetector,
    ControllerTestHarness,
    FakeNodeManager,
    FakeNotifier,
    FakeTrainingJob,
    FixedDecisionDetector,
    make_test_controller,
)
from tests.fast.utils.ft.helpers.diagnostic_fakes import (
    FakeDiagnosticScheduler,
    FakeNodeAgent,
    SlowDiagnostic,
    StubDiagnostic,
    make_fake_agents,
    mock_inter_machine_run,
    mock_stack_trace_diagnostic,
)
from tests.fast.utils.ft.helpers.metric_injectors import (
    EMPTY_RANK_PLACEMENT,
    get_sample_value,
    inject_critical_xid,
    inject_disk_fault,
    inject_gpu_temperature,
    inject_gpu_unavailable,
    inject_healthy_node,
    inject_nic_down,
    inject_nic_up,
    inject_training_job_status,
    make_detector_context,
    make_fake_metric_store,
    make_fake_mini_wandb,
    make_metric,
    make_test_exporter,
)

__all__ = [
    "EMPTY_RANK_PLACEMENT",
    "SAMPLE_PYSPY_OUTPUT_DIFFERENT_STUCK",
    "SAMPLE_PYSPY_OUTPUT_NORMAL",
    "SAMPLE_PYSPY_OUTPUT_STUCK",
    "AlwaysMarkBadDetector",
    "AlwaysNoneDetector",
    "ControllerTestHarness",
    "FakeDiagnosticScheduler",
    "FakeKmsgReader",
    "FakeNodeAgent",
    "FakeNodeManager",
    "FakeNotifier",
    "FakeTrainingJob",
    "FixedDecisionDetector",
    "SlowDiagnostic",
    "StubDiagnostic",
    "TestCollector",
    "create_sysfs_interface",
    "get_sample_value",
    "inject_critical_xid",
    "inject_disk_fault",
    "inject_gpu_temperature",
    "inject_gpu_unavailable",
    "inject_healthy_node",
    "inject_nic_down",
    "inject_nic_up",
    "inject_training_job_status",
    "make_detector_context",
    "make_fake_agents",
    "make_fake_metric_store",
    "make_fake_mini_wandb",
    "make_metric",
    "make_mock_pynvml",
    "make_rank_pids_provider",
    "make_test_controller",
    "make_test_exporter",
    "make_trace_result",
    "mock_inter_machine_run",
    "mock_stack_trace_diagnostic",
]
