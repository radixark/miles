from tests.utils.soak.core.events import SoakEvent
from tests.utils.soak.core.types import SoakForms

MIN_SOAK_INJECTIONS: int = 2


def assert_healing(
    ft_components: tuple[str, ...],
    *,
    events: list[SoakEvent],
    forms: SoakForms,
    context: str,
) -> None:
    raise NotImplementedError


def assert_min_injections(events: list[SoakEvent], *, kind: str, context: str) -> None:
    raise NotImplementedError


def assert_injections_recovered(events: list[SoakEvent], *, cell_type: str, forms: SoakForms) -> None:
    raise NotImplementedError
