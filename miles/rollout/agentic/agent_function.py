"""The agent-function contract for ``agentic_tool_call.generate``.

An agent function (``--custom-agent-function-path``) drives one episode against
the session-scoped policy URL and returns a dict to merge into the sample's
metadata, or None when there is nothing to add. Either way the recorded session
becomes a training sample. The one way to discard the sample is to raise
:class:`InfraAbort`.

This module must stay free of heavy imports: agent functions import it on
CPU-only hosts and in offline tests where torch is not available.
"""


class InfraAbort(Exception):
    """Raised by an agent function to discard the sample it is producing.

    Reserve it for failures the policy cannot have caused: the sandbox platform
    refusing to create a sandbox, an environment host process that died, the
    trainer losing its network path to the environment. A discarded sample
    contributes no gradient, so any outcome the policy CAN bring about -- a
    timeout, a sandbox the agent broke, a verifier the agent starved -- must be
    returned as reward 0 instead; discarding those teaches the policy to trigger
    them to escape the penalty.

    ``exit_status`` names the cause. It is recorded on the aborted sample's
    metadata and counted in the ``rollout/aborted/drop_<exit_status>`` metric.
    """

    def __init__(self, exit_status: str, message: str | None = None):
        super().__init__(message or exit_status)
        self.exit_status = exit_status
