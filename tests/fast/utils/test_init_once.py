import pytest

from miles.utils.init_once import InitOnce, InitState, init_once

pytestmark = pytest.mark.asyncio


class _Component:
    def __init__(self) -> None:
        self._init_once = InitOnce(type(self).__name__)
        self.runs = 0

    @init_once
    def init(self, *, fail: bool = False) -> str:
        self.runs += 1
        if fail:
            raise ValueError("init blew up")
        return "built"

    def is_initialized(self) -> bool:
        return self._init_once.is_initialized()


class _AsyncComponent:
    def __init__(self) -> None:
        self._init_once = InitOnce(type(self).__name__)
        self.runs = 0

    @init_once
    async def init(self, *, fail: bool = False) -> str:
        self.runs += 1
        if fail:
            raise ValueError("init blew up")
        return "built"

    def is_initialized(self) -> bool:
        return self._init_once.is_initialized()


class TestTheStateAComponentReports:
    def test_a_component_that_never_ran_init_is_not_initialized(self):
        """A component only answers yes once it finished building, never because it was constructed."""
        assert _Component().is_initialized() is False

    def test_a_component_that_finished_init_is_initialized(self):
        """This is the answer a take-over reads to decide between building and resuming."""
        component = _Component()

        component.init()

        assert component.is_initialized() is True

    def test_a_component_whose_init_raised_is_not_initialized(self):
        """A half-built component taken over as a built one would run with a fleet nobody finished wiring."""
        component = _Component()

        with pytest.raises(ValueError):
            component.init(fail=True)

        assert component.is_initialized() is False
        assert component._init_once.state is InitState.INIT_FAILED

    def test_a_component_inside_its_own_init_is_not_initialized(self):
        """The window between entering init and finishing it must never read as built."""
        guard = InitOnce("Component")

        with guard.guarding():
            assert guard.is_initialized() is False
            assert guard.state is InitState.INITIALIZING


class TestInitRunsExactlyOnce:
    def test_a_second_init_is_refused(self):
        """Re-initializing a live component would throw away the state it is holding."""
        component = _Component()
        component.init()

        with pytest.raises(AssertionError, match="stale worker"):
            component.init()

        assert component.runs == 1

    def test_a_second_init_is_refused_after_the_first_one_raised(self):
        """Nothing can rebuild a process that already ran half of an init over its own state."""
        component = _Component()
        with pytest.raises(ValueError):
            component.init(fail=True)

        with pytest.raises(AssertionError, match="stale worker"):
            component.init()

        assert component.runs == 1

    def test_the_refusal_names_the_state_it_found(self):
        """An operator reading the crash has to be able to tell a finished init from a failed one."""
        component = _Component()
        with pytest.raises(ValueError):
            component.init(fail=True)

        with pytest.raises(AssertionError, match="init_failed"):
            component.init()


class TestTheGuardOnAsyncInit:
    async def test_an_async_init_that_finished_reports_initialized(self):
        """Three of the four components this guards await their init."""
        component = _AsyncComponent()

        assert await component.init() == "built"
        assert component.is_initialized() is True

    async def test_an_async_init_that_raised_reports_uninitialized(self):
        """The failure path has to hold for the awaited form too."""
        component = _AsyncComponent()

        with pytest.raises(ValueError):
            await component.init(fail=True)

        assert component.is_initialized() is False

    async def test_a_second_async_init_is_refused(self):
        """The exactly-once refusal is what catches an operator driving a live fleet by hand."""
        component = _AsyncComponent()
        await component.init()

        with pytest.raises(AssertionError, match="stale worker"):
            await component.init()
