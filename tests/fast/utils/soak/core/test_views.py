from tests.fast.utils.soak.soak_fakes import _applied, _at, _cell_target, _request, _requested, _result
from tests.utils.soak.core.views import project_actions


class TestProjectActions:
    def test_requested_applied_and_result_are_joined_by_request_id(self) -> None:
        """Each request collects its own effect and result, never another request's."""
        first = _request(_cell_target(), request_id="first")
        second = _request(_cell_target(cell_index=1), request_id="second")
        events = [
            _requested(first, at=_at(0)),
            _requested(second, at=_at(1)),
            _applied(second, at=_at(2)),
            _result(first, at=_at(3), returned=False),
        ]

        actions = project_actions(events)

        assert list(actions) == ["first", "second"]
        assert (actions["first"].applied, actions["first"].result.returned) == (None, False)
        assert (actions["second"].applied.request_id, actions["second"].result) == ("second", None)

    def test_an_effect_without_a_request_is_not_an_action(self) -> None:
        """Applied or result events of unknown requests do not create actions."""
        stray = _request(_cell_target(), request_id="stray")

        assert project_actions([_applied(stray, at=_at(0)), _result(stray, at=_at(1))]) == {}
