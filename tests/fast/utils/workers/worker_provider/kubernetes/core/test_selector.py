from __future__ import annotations

from miles.utils.workers.worker_provider.kubernetes.core import provider
from miles.utils.workers.worker_provider.kubernetes.helm import env

BASE_SELECTOR = "app.kubernetes.io/instance=r"


class TestWatchedPodsSelector:
    def test_watches_nothing_when_no_pool_is_wanted(self) -> None:
        """A run with no dynamic pool must watch zero pods, so the selector names a label no pod carries."""
        selector = provider._watched_pods_selector(
            base_selector=BASE_SELECTOR, pool_label_key=env.DEFAULT_LABEL_KEYS.pool_id, pool_ids=[]
        )

        assert selector == f"{BASE_SELECTOR},{provider._NO_POD_CARRIES_THIS_LABEL}"

    def test_scopes_the_watch_to_the_wanted_pools(self) -> None:
        """Pods of other pools in the same release must stay outside the watch."""
        selector = provider._watched_pods_selector(
            base_selector=BASE_SELECTOR, pool_label_key=env.DEFAULT_LABEL_KEYS.pool_id, pool_ids=["engine"]
        )

        assert selector == f"{BASE_SELECTOR},{env.DEFAULT_LABEL_KEYS.pool_id} in (engine)"

    def test_sorts_the_wanted_pools(self) -> None:
        """The selector is compared and logged, so the same pool set must always spell the same string."""
        selector = provider._watched_pods_selector(
            base_selector=BASE_SELECTOR,
            pool_label_key=env.DEFAULT_LABEL_KEYS.pool_id,
            pool_ids=["rollout", "engine"],
        )

        assert selector == f"{BASE_SELECTOR},{env.DEFAULT_LABEL_KEYS.pool_id} in (engine,rollout)"

    def test_keeps_the_base_selector_in_front(self) -> None:
        """The release selector is what keeps one run from watching another run's pods."""
        selector = provider._watched_pods_selector(
            base_selector="app=miles", pool_label_key="pool", pool_ids=["engine"]
        )

        assert selector.startswith("app=miles,")
