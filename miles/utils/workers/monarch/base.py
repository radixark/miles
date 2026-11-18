"""The base class of every Monarch actor Miles defines."""

import logging

from monarch.actor import Actor, InvalidReferenceReason, UndeliverableMessageEnvelope

logger = logging.getLogger(__name__)

# Calls whose destination stopped or failed. Any other invalid reference (a handler never bound, a protocol
# mismatch, ...) is a bug and stays fatal.
_GONE_DESTINATION_REASONS = (InvalidReferenceReason.ActorStopped, InvalidReferenceReason.ActorFailed)


class MilesActor(Actor):
    """Base of every Monarch actor Miles defines: holds the behaviour they all share."""

    def _handle_invalid_reference(self, message: UndeliverableMessageEnvelope) -> bool:
        """Keep this actor alive when a message it sent reaches an actor that has stopped or failed.

        Monarch fails the call itself with SupervisionError, and by default also fails the actor that sent it.
        Miles actors routinely message actors that are going away: the worker manager's liveness scan pings
        worker meshes whose actors are stopping, e.g. while a run shuts down, and a worker calls actors that can
        stop before it. The default would bring down the run's manager or one of its workers over a destination
        that is already gone. The call still raises in its caller, which is how the liveness scan finds the cell
        dead, so absorbing the failure here loses nothing. Any other invalid reference is a bug and stays fatal.
        """
        failure = message.root_delivery_failure()
        if failure is None or failure.invalid_reference_reason not in _GONE_DESTINATION_REASONS:
            return False
        logger.info(f"Dropped the delivery failure of a message to an actor that is gone: {failure}")
        return True
