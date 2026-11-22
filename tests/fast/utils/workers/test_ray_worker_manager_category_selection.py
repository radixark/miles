import pytest
from tests.fast.utils.workers.fake_ray import FakeRayCluster
from tests.fast.utils.workers.test_ray_worker_manager import _launch, _make_spec


class TestCategorySelection:
    async def test_explicit_pools_are_intersected_with_the_requested_category(
        self, fake_ray_cluster: FakeRayCluster
    ) -> None:
        """A category filter cannot expose an unrequested pool with the same category."""
        specs = [
            _make_spec("west").model_copy(update={"category": "inference-engine"}),
            _make_spec("east").model_copy(update={"category": "inference-engine"}),
            _make_spec("trainer").model_copy(update={"category": "trainer-engine"}),
        ]
        manager = await _launch(specs=specs)

        assert sorted(manager.get_cell_infos(pool_ids=["west", "trainer"], category="inference-engine")) == [
            "west-00000"
        ]
        assert manager.get_cell_infos(pool_ids=[], category="inference-engine") == {}

    async def test_a_category_filter_does_not_hide_an_unknown_explicit_pool(
        self, fake_ray_cluster: FakeRayCluster
    ) -> None:
        """A misspelled pool remains an error even when no cell matches the category."""
        manager = await _launch(specs=[_make_spec("engine")])

        with pytest.raises(AssertionError, match="missing"):
            manager.get_cell_infos(pool_ids=["missing"], category="no-matching-category")
