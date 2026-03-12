"""Node-side diagnostic executors.

Each executor implements ``NodeExecutorProtocol`` and runs locally on a single
node (e.g. GPU health check, NCCL test, stack-trace collection).  Results are
``DiagnosticResult`` objects returned to the cluster-side orchestrator.
"""
