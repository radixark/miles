import pytest
from tests.fast.utils.soak.soak_fakes import (
    _at,
    _cell_target,
    _FakeForm,
    _healed_injection,
    _injected,
    _observation,
    _reconfigure,
    _request,
    _requested,
    _result,
    _step_end,
)
from tests.utils.soak.core.events import SoakEvent
from tests.utils.soak.core.types import SoakForms
from tests.utils.soak.ft.actions.inject_fault import InjectFaultForm
from tests.utils.soak.ft.checkers.healing import assert_healing

from miles.utils.test_utils.fault_injector.actions.process import ExitProcessAction, KillProcessAction

_KILL = InjectFaultForm(base_url="http://api:18080", action=KillProcessAction())
_EXIT = InjectFaultForm(base_url="http://api:18080", action=ExitProcessAction())


def _forms(**extra: list) -> SoakForms:
    return {"actor": [_KILL], "rollout": [_KILL], **extra}


def _healed(kind: str, cell_index: int, start: float, **kwargs: object) -> list[SoakEvent]:
    form_name = kwargs.pop("form_name", _KILL.name)
    return _healed_injection(
        form_name, kind=kind, cell_index=cell_index, start=start, request_id=f"{kind}-{start}", **kwargs
    )


class TestTrainerHealing:
    def test_every_injection_healed_by_its_own_cell_passes(self) -> None:
        """The checker stays silent on the path a healthy trainer soak takes."""
        assert_healing(
            ("train",), events=[*_healed("actor", 1, 0), *_healed("actor", 2, 10)], forms=_forms(), context="t"
        )

    def test_fewer_than_two_injections_prove_too_little(self) -> None:
        """One healed trainer fault does not exercise recovery more than once."""
        with pytest.raises(AssertionError, match="only 1 successful injection"):
            assert_healing(("train",), events=_healed("actor", 1, 0), forms=_forms(), context="t")

    def test_rollout_injections_do_not_count_toward_the_trainer_floor(self) -> None:
        """A mixed soak's engine crashes say nothing about trainer healing."""
        events = [*_healed("actor", 1, 0), *_healed("rollout", 0, 10), *_healed("rollout", 1, 20)]

        with pytest.raises(AssertionError, match="trainer cells"):
            assert_healing(("train",), events=events, forms=_forms(), context="t")

    def test_requests_that_never_applied_do_not_count_toward_the_floor(self) -> None:
        """A draw that never landed is not an injection."""
        refused = _request(_cell_target(cell_index=2), form_name=_KILL.name, request_id="refused")
        events = [*_healed("actor", 1, 0), _requested(refused, at=_at(10)), _result(refused, at=_at(11))]

        with pytest.raises(AssertionError, match="only 1 successful injection"):
            assert_healing(("train",), events=events, forms=_forms(), context="t")

    def test_a_final_injection_that_never_healed_fails_although_the_floor_is_cleared(self) -> None:
        """Three landed faults with two heals leave the run degraded and must fail."""
        events = [*_healed("actor", 1, 0), *_healed("actor", 2, 10), *_healed("actor", 1, 20, ready=False)]

        with pytest.raises(AssertionError, match="recovery witness failed"):
            assert_healing(("train",), events=events, forms=_forms(), context="t")

    def test_a_trainer_back_without_a_reconfigure_witness_is_not_healed(self) -> None:
        """A restarted trainer never readmitted into the group has not healed."""
        events = [*_healed("actor", 1, 0), *_healed("actor", 2, 10, healed_cell_indices=[])]

        with pytest.raises(AssertionError, match="recovery witness failed"):
            assert_healing(("train",), events=events, forms=_forms(), context="t")

    def test_healing_a_cell_that_was_never_injected_does_not_pay_another_cells_debt(self) -> None:
        """Healings are paired with the injected cell index, not merely counted."""
        events = [*_healed("actor", 1, 0), *_healed("actor", 2, 10, healed_cell_indices=[3])]

        with pytest.raises(AssertionError, match="recovery witness failed"):
            assert_healing(("train",), events=events, forms=_forms(), context="t")

    def test_two_cells_healed_by_one_reconfigure_both_count(self) -> None:
        """One reconfigure readmitting two injected cells heals both of them."""
        first = _request(_cell_target(cell_index=1), form_name=_KILL.name, request_id="first")
        second = _request(_cell_target(cell_index=2), form_name=_KILL.name, request_id="second")
        healing = _reconfigure(
            at=_at(6),
            healed_cell_indices=[1, 2],
            cell_incarnations_after={first.target.identity: "inc-b", second.target.identity: "inc-c"},
        )
        events = [
            *_injected(first, start=0),
            *_injected(second, start=3),
            _observation(
                [
                    first.target.model_copy(update={"incarnation": "inc-b"}),
                    second.target.model_copy(update={"incarnation": "inc-c"}),
                ],
                at=_at(7),
                new_sut_events=[healing],
            ),
            _observation(None, at=_at(8), new_sut_events=[_step_end(3, at=_at(8))]),
        ]

        assert_healing(("train",), events=events, forms=_forms(), context="t")


class TestRolloutHealing:
    def test_a_fresh_serve_after_every_injection_passes(self) -> None:
        """Rollout cells heal through a ready new incarnation without any trainer reconfigure."""
        assert_healing(
            ("rollout",), events=[*_healed("rollout", 0, 0), *_healed("rollout", 1, 10)], forms=_forms(), context="r"
        )

    def test_a_last_victim_still_relaunching_fails(self) -> None:
        """A rollout soak ending with its last victim never ready again must fail."""
        events = [*_healed("rollout", 0, 0), *_healed("rollout", 1, 10, ready=False)]

        with pytest.raises(AssertionError, match="rollout recovery witness failed"):
            assert_healing(("rollout",), events=events, forms=_forms(), context="r")

    def test_a_ready_reading_of_the_killed_incarnation_is_not_healing(self) -> None:
        """A stale serve of the dead incarnation cannot clear the injected cell."""
        events = [*_healed("rollout", 0, 0), *_healed("rollout", 1, 10, new_incarnation="inc-a")]

        with pytest.raises(AssertionError, match="rollout recovery witness failed"):
            assert_healing(("rollout",), events=events, forms=_forms(), context="r")

    def test_a_siblings_fresh_serve_does_not_clear_the_injected_cell(self) -> None:
        """Only the injected cell's own replacement discharges its injection."""
        victim = _request(_cell_target(kind="rollout", cell_index=1), form_name=_KILL.name, request_id="victim")
        sibling = _cell_target(kind="rollout", cell_index=2, incarnation="inc-b")
        events = [
            *_healed("rollout", 0, 0),
            *_injected(victim, start=10),
            _observation([sibling], at=_at(13)),
            _observation(None, at=_at(14), new_sut_events=[_step_end(9, at=_at(14))]),
        ]

        with pytest.raises(AssertionError, match="rollout recovery witness failed"):
            assert_healing(("rollout",), events=events, forms=_forms(), context="r")


class TestEnabledFormsWorked:
    def test_an_enabled_form_that_never_landed_fails_the_soak(self) -> None:
        """Clearing the floor with one form must not hide another form never landing."""
        events = [*_healed("actor", 1, 0), *_healed("actor", 2, 10)]

        with pytest.raises(AssertionError, match="never injected successfully"):
            assert_healing(("train",), events=events, forms=_forms(actor=[_KILL, _EXIT]), context="t")

    def test_every_enabled_form_landing_passes(self) -> None:
        """Each form landing at least once satisfies the form check."""
        events = [*_healed("actor", 1, 0), *_healed("actor", 2, 10, form_name=_EXIT.name)]

        assert_healing(("train",), events=events, forms=_forms(actor=[_KILL, _EXIT]), context="t")

    def test_forms_of_a_component_without_ft_are_not_required(self) -> None:
        """A trainer-only soak is not failed for engines it was told to leave alone."""
        events = [*_healed("actor", 1, 0), *_healed("actor", 2, 10)]

        assert_healing(("train",), events=events, forms=_forms(rollout=[_KILL, _EXIT]), context="t")

    def test_a_harmless_form_is_not_held_to_a_recovery_witness(self) -> None:
        """A form that leaves its cell running has nothing to heal, but still counts as landed."""
        probe = _FakeForm(name="probe", harms_target=False, recovered=False)
        events = [*_healed("actor", 1, 0), *_healed("actor", 2, 10), *_healed("actor", 1, 20, form_name="probe")]

        assert_healing(("train",), events=events, forms=_forms(actor=[_KILL, probe]), context="t")
