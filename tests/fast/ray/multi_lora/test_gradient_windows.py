from miles.ray.multi_lora.gradient_windows import GradientWindowTracker

KEY_A = ("A", "reg-1")
KEY_A2 = ("A", "reg-2")
KEY_B = ("B", "reg-1")


class TestDirtyFlag:
    def test_successful_fb_sets_dirty_and_forward_never_calls_in(self):
        tracker = GradientWindowTracker()
        tracker.open(KEY_A)
        assert not tracker.is_dirty(KEY_A)
        tracker.mark_forward_backward_succeeded(KEY_A)
        assert tracker.is_dirty(KEY_A)

    def test_committed_step_consumes_the_window(self):
        tracker = GradientWindowTracker()
        tracker.mark_forward_backward_succeeded(KEY_A)
        assert tracker.commit_step(KEY_A) == 1
        assert not tracker.is_dirty(KEY_A)
        assert tracker.step_of(KEY_A) == 1

    def test_executed_optim_without_commit_clears_without_advancing(self):
        tracker = GradientWindowTracker()
        tracker.mark_forward_backward_succeeded(KEY_A)
        tracker.clear_after_executed_optim(KEY_A)
        assert not tracker.is_dirty(KEY_A)
        assert tracker.step_of(KEY_A) == 0

    def test_clean_commit_needs_no_prior_fb(self):
        tracker = GradientWindowTracker()
        assert tracker.commit_step(KEY_A) == 1


class TestStreamIdentity:
    def test_registrations_of_the_same_name_are_different_streams(self):
        tracker = GradientWindowTracker()
        tracker.mark_forward_backward_succeeded(KEY_A)
        assert not tracker.is_dirty(KEY_A2)
        assert tracker.commit_step(KEY_A2) == 1
        assert tracker.is_dirty(KEY_A)

    def test_streams_are_independent_across_names(self):
        tracker = GradientWindowTracker()
        tracker.commit_step(KEY_A)
        tracker.commit_step(KEY_A)
        tracker.mark_forward_backward_succeeded(KEY_B)
        assert tracker.step_of(KEY_A) == 2 and not tracker.is_dirty(KEY_A)
        assert tracker.step_of(KEY_B) == 0 and tracker.is_dirty(KEY_B)

    def test_close_drops_the_stream_and_queries_go_inert(self):
        tracker = GradientWindowTracker()
        tracker.commit_step(KEY_A)
        tracker.mark_forward_backward_succeeded(KEY_A)
        tracker.close(KEY_A)
        assert tracker.step_of(KEY_A) == 0
        assert not tracker.is_dirty(KEY_A)


class TestRestore:
    def test_restore_moves_the_clock(self):
        tracker = GradientWindowTracker()
        tracker.restore_step(KEY_A, 42)
        assert tracker.step_of(KEY_A) == 42
        assert tracker.commit_step(KEY_A) == 43
