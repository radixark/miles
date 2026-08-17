"""Registration-keyed gradient-window state for the tinker backend.

Parameterization-neutral (codex-rollout-fullparameter-design-0810 §3.4): a
training stream is identified by its ``RegistrationKey`` (adapter name,
registration id) — no slots, no residency, no Multi-LoRA imports. The tracker
is the authority for each live stream's step clock and dirty flag (unstepped
accumulated gradients that no checkpoint carries). Two things it deliberately
does NOT own:

- Poison evidence: ``OperationLedger.poisoned_window_blocker()`` stays the
  sole authority; no second poison history lives here.
- Multi-LoRA lifecycle: the ``AdapterRegistry`` mirrors dirty transitions into
  its SlotPool pins and reacts to committed steps (``num_step`` auto-retire)
  through hooks — the pin is a residency-side mirror, never the protocol
  state's only storage.

A future full-parameter backend can reuse this stream state without ever
constructing a SlotPool; a future paging policy may query ``is_dirty()`` per
registration, but eviction policy is explicitly out of scope here.
"""

from dataclasses import dataclass

from miles.utils.tinker_backend import RegistrationKey


@dataclass
class TrainingStreamState:
    step: int = 0
    # True while the stream holds unstepped accumulated gradients.
    dirty: bool = False


class GradientWindowTracker:
    """Step/dirty authority for every live training stream."""

    def __init__(self) -> None:
        self._streams: dict[RegistrationKey, TrainingStreamState] = {}

    def _stream(self, key: RegistrationKey) -> TrainingStreamState:
        return self._streams.setdefault(key, TrainingStreamState())

    # ------------------------------ lifecycle ------------------------------

    def open(self, key: RegistrationKey) -> None:
        """Start tracking a registration's stream (idempotent)."""
        self._stream(key)

    def close(self, key: RegistrationKey) -> None:
        """Drop a retired registration's stream state."""
        self._streams.pop(key, None)

    # ------------------------------ queries ------------------------------

    def step_of(self, key: RegistrationKey) -> int:
        stream = self._streams.get(key)
        return stream.step if stream is not None else 0

    def is_dirty(self, key: RegistrationKey) -> bool:
        stream = self._streams.get(key)
        return stream is not None and stream.dirty

    # ------------------------------ transitions ------------------------------

    def mark_forward_backward_succeeded(self, key: RegistrationKey) -> None:
        """A forward_backward landed: the stream holds unstepped gradients.
        (A plain forward never calls this — it produces no gradient.)"""
        self._stream(key).dirty = True

    def clear_after_executed_optim(self, key: RegistrationKey) -> None:
        """An optim_step EXECUTED without committing a step (veto or poison
        discard): the window's gradients were cleared on every rank, so the
        stream is clean, but the step clock never moves."""
        self._stream(key).dirty = False

    def commit_step(self, key: RegistrationKey) -> int:
        """A successful optim_step consumed the window: advance the step clock,
        clear the dirty flag, and return the committed step."""
        stream = self._stream(key)
        stream.step += 1
        stream.dirty = False
        return stream.step

    def restore_step(self, key: RegistrationKey, step: int) -> None:
        """A load_state (or registration resume) repositioned the stream's
        clock. The num_step baseline (``start_step``) is the registry's
        authority — the tracker keeps no duplicate copy of it."""
        stream = self._stream(key)
        stream.step = step
