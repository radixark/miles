HOT_RESTART_FORM_NAME: str = "hot_restart"
TAKE_OVER_TIMEOUT_SECONDS: float = 1800.0
TAKE_OVER_POLL_INTERVAL_SECONDS: float = 10.0


def restamped_replaced_workloads(
    *, before: dict[str, str | None], after: dict[str, str | None] | None, workloads: frozenset[str]
) -> bool:
    if after is None:
        return False
    return all((stamp := after.get(one)) is not None and stamp != before.get(one) for one in workloads)
