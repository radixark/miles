"""Controller tests — split into focused modules.

This file re-exports all test classes so that any runner or import
targeting ``test_controller`` still finds them.

Focused modules:
- test_controller_register_rank: registration, rank PIDs, log_step
- test_controller_execute_decision: detector chain, decision dispatch, mark-bad
- test_controller_recovery_mode: enter/exit recovery, exporter mode
- test_controller_status: status dict, exporter gauges, shutdown, agent mgmt
"""
from tests.fast.utils.ft.test_controller_register_rank import (  # noqa: F401
    TestGetRankPidsForNode,
    TestLogStep,
    TestRegisterRank,
)
from tests.fast.utils.ft.test_controller_execute_decision import (  # noqa: F401
    TestDetectorChain,
    TestExecuteDecision,
    TestMarkBadAndRestartReal,
    TestTickEmptyDetectorChain,
)
from tests.fast.utils.ft.test_controller_recovery_mode import (  # noqa: F401
    TestEnterRecovery,
)
from tests.fast.utils.ft.test_controller_status import (  # noqa: F401
    TestAgentManagement,
    TestDefaultDiagnosticPipeline,
    TestDefaultDiagnosticSchedulerWiring,
    TestGetStatus,
    TestScrapeLoopDefensiveBranches,
    TestShutdown,
    TestTrainingJobStatusExporter,
)
