"""Cluster-side diagnostic orchestrators.

Each orchestrator implements ``ClusterExecutorProtocol`` and coordinates
diagnostics across multiple nodes (e.g. neighbor NCCL, per-node dispatch).
They call into node agents via ``NodeAgentProtocol.run_diagnostic()``.
"""
