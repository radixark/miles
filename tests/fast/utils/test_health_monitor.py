from miles.utils.health_monitor import compute_kill_list


class TestComputeKillListBasic:
    def test_empty_failures(self):
        assert compute_kill_list([], total_engines=8, max_kill_ratio=0.5) == []

    def test_single_failure(self):
        result = compute_kill_list([3], total_engines=8, max_kill_ratio=0.5)
        assert result == [3]

    def test_multiple_failures(self):
        result = compute_kill_list([1, 5], total_engines=8, max_kill_ratio=0.5)
        assert result == [1, 5]

    def test_result_is_sorted(self):
        result = compute_kill_list([7, 2, 5], total_engines=8, max_kill_ratio=1.0)
        assert result == [2, 5, 7]

    def test_deduplicates(self):
        result = compute_kill_list([3, 3, 1, 1], total_engines=8, max_kill_ratio=1.0)
        assert result == [1, 3]


class TestAntiCascade:
    def test_kills_capped_at_max_ratio(self):
        """With max_kill_ratio=0.5 and 8 engines, at most 4 can be killed per round."""
        failed = [0, 1, 2, 3, 4, 5]
        result = compute_kill_list(failed, total_engines=8, max_kill_ratio=0.5)
        assert len(result) == 4
        assert result == [0, 1, 2, 3]

    def test_at_least_one_kill_even_with_small_ratio(self):
        result = compute_kill_list([0], total_engines=8, max_kill_ratio=0.01)
        assert result == [0]

    def test_all_killed_when_ratio_is_one(self):
        failed = list(range(8))
        result = compute_kill_list(failed, total_engines=8, max_kill_ratio=1.0)
        assert result == list(range(8))

    def test_small_cluster_two_engines(self):
        result = compute_kill_list([0, 1], total_engines=2, max_kill_ratio=0.5)
        assert len(result) == 1
