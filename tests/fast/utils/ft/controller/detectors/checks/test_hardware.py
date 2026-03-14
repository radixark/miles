"""Tests for hardware_checks edge cases and boundary conditions."""

from datetime import timedelta

from tests.fast.utils.ft.conftest import inject_disk_fault, inject_nic_down, inject_nic_up, make_fake_metric_store

from miles.utils.ft.agents.types import GaugeSample
from miles.utils.ft.controller.detectors.checks.hardware import (
    _analyze_nic_flap_transitions,
    _analyze_nic_persistent_down,
    _query_nic_timeseries,
    check_all_nic_faults,
    check_disk_fault,
    check_majority_nic_down,
)
from miles.utils.ft.utils.metric_names import NODE_NETWORK_UP

WINDOW = timedelta(minutes=5)


def _build_nic_df(store, window=WINDOW):
    return _query_nic_timeseries(store, window)


# ---------------------------------------------------------------------------
# _analyze_nic_flap_transitions
# ---------------------------------------------------------------------------


class TestAnalyzeNicFlapTransitions:
    def test_consecutive_down_samples_are_single_transition(self) -> None:
        """Multiple consecutive down samples after one up should count as 1 transition."""
        store = make_fake_metric_store()
        inject_nic_up(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib0")

        df = _build_nic_df(store)
        faults = _analyze_nic_flap_transitions(df, threshold=1, window=WINDOW)

        assert len(faults) == 1
        assert "1 time(s)" in faults[0].reason

    def test_no_preceding_up_means_no_transition(self) -> None:
        """Down samples without a preceding up sample in the window are not transitions."""
        store = make_fake_metric_store()
        inject_nic_down(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib0")

        df = _build_nic_df(store)
        faults = _analyze_nic_flap_transitions(df, threshold=1, window=WINDOW)

        assert faults == []

    def test_all_samples_up_no_transition(self) -> None:
        store = make_fake_metric_store()
        inject_nic_up(store, node_id="node-0", device="ib0")
        inject_nic_up(store, node_id="node-0", device="ib0")
        inject_nic_up(store, node_id="node-0", device="ib0")

        df = _build_nic_df(store)
        faults = _analyze_nic_flap_transitions(df, threshold=1, window=WINDOW)

        assert faults == []

    # --- Multi-device / multi-node aggregation ---

    def test_multiple_devices_transitions_summed_per_node(self) -> None:
        """Transitions across different devices on the same node sum together."""
        store = make_fake_metric_store()
        inject_nic_up(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib0")
        inject_nic_up(store, node_id="node-0", device="ib1")
        inject_nic_down(store, node_id="node-0", device="ib1")

        df = _build_nic_df(store)
        faults = _analyze_nic_flap_transitions(df, threshold=2, window=WINDOW)

        assert len(faults) == 1
        assert faults[0].node_id == "node-0"

    def test_multiple_devices_independent_transition_tracking(self) -> None:
        """Each device tracks its own transition state independently."""
        store = make_fake_metric_store()
        # ib0: up→down→up→down = 2 transitions
        inject_nic_up(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib0")
        inject_nic_up(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib0")
        # ib1: up→down = 1 transition
        inject_nic_up(store, node_id="node-0", device="ib1")
        inject_nic_down(store, node_id="node-0", device="ib1")

        df = _build_nic_df(store)
        faults = _analyze_nic_flap_transitions(df, threshold=3, window=WINDOW)

        assert len(faults) == 1
        assert "3 time(s)" in faults[0].reason

    def test_multiple_nodes_independent(self) -> None:
        """Different nodes have independent transition counts."""
        store = make_fake_metric_store()
        # node-0: 1 transition
        inject_nic_up(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib0")
        # node-1: 2 transitions
        inject_nic_up(store, node_id="node-1", device="ib0")
        inject_nic_down(store, node_id="node-1", device="ib0")
        inject_nic_up(store, node_id="node-1", device="ib0")
        inject_nic_down(store, node_id="node-1", device="ib0")

        df = _build_nic_df(store)
        faults = _analyze_nic_flap_transitions(df, threshold=2, window=WINDOW)

        assert len(faults) == 1
        assert faults[0].node_id == "node-1"

    # --- Threshold boundaries ---

    def test_exactly_at_threshold_triggers(self) -> None:
        store = make_fake_metric_store()
        inject_nic_up(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib0")
        inject_nic_up(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib0")

        df = _build_nic_df(store)
        faults = _analyze_nic_flap_transitions(df, threshold=2, window=WINDOW)

        assert len(faults) == 1

    def test_below_threshold_no_fault(self) -> None:
        store = make_fake_metric_store()
        inject_nic_up(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib0")

        df = _build_nic_df(store)
        faults = _analyze_nic_flap_transitions(df, threshold=2, window=WINDOW)

        assert faults == []

    def test_threshold_one_single_transition_triggers(self) -> None:
        store = make_fake_metric_store()
        inject_nic_up(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib0")

        df = _build_nic_df(store)
        faults = _analyze_nic_flap_transitions(df, threshold=1, window=WINDOW)

        assert len(faults) == 1

    # --- Complex sequences ---

    def test_rapid_flapping(self) -> None:
        """Rapid up/down/up/down/up/down counts 3 transitions."""
        store = make_fake_metric_store()
        inject_nic_up(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib0")
        inject_nic_up(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib0")
        inject_nic_up(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib0")

        df = _build_nic_df(store)
        faults = _analyze_nic_flap_transitions(df, threshold=3, window=WINDOW)

        assert len(faults) == 1
        assert "3 time(s)" in faults[0].reason

    def test_down_up_down_is_one_transition_not_two(self) -> None:
        """down→up is not a transition; only up→down counts."""
        store = make_fake_metric_store()
        inject_nic_down(store, node_id="node-0", device="ib0")
        inject_nic_up(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib0")

        df = _build_nic_df(store)
        faults = _analyze_nic_flap_transitions(df, threshold=1, window=WINDOW)

        assert len(faults) == 1
        assert "1 time(s)" in faults[0].reason

    def test_reason_message_contains_transition_count(self) -> None:
        store = make_fake_metric_store()
        inject_nic_up(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib0")
        inject_nic_up(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib0")

        df = _build_nic_df(store)
        faults = _analyze_nic_flap_transitions(df, threshold=1, window=WINDOW)

        assert len(faults) == 1
        assert "2 time(s)" in faults[0].reason
        assert "went down" in faults[0].reason


# ---------------------------------------------------------------------------
# _analyze_nic_persistent_down
# ---------------------------------------------------------------------------


class TestAnalyzeNicPersistentDown:
    def test_nic_went_up_then_stayed_down(self) -> None:
        """Classic permanent crash: was up, then went down and never recovered."""
        store = make_fake_metric_store()
        inject_nic_up(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib0")

        df = _build_nic_df(store)
        faults = _analyze_nic_persistent_down(df)

        assert len(faults) == 1
        assert faults[0].node_id == "node-0"
        assert "persistently down" in faults[0].reason
        assert "ib0" in faults[0].reason

    def test_always_down_no_fault(self) -> None:
        """NIC that was never up: no had_up → not a crash, just never started."""
        store = make_fake_metric_store()
        inject_nic_down(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib0")

        df = _build_nic_df(store)
        faults = _analyze_nic_persistent_down(df)

        assert faults == []

    def test_single_sample_not_enough(self) -> None:
        """Need at least 2 samples (sample_count >= 2) to confirm a transition."""
        store = make_fake_metric_store()
        inject_nic_down(store, node_id="node-0", device="ib0")

        df = _build_nic_df(store)
        faults = _analyze_nic_persistent_down(df)

        assert faults == []

    def test_recovered_nic_no_fault(self) -> None:
        """NIC went down but came back up: last_value > 0 → not persistent."""
        store = make_fake_metric_store()
        inject_nic_up(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib0")
        inject_nic_up(store, node_id="node-0", device="ib0")

        df = _build_nic_df(store)
        faults = _analyze_nic_persistent_down(df)

        assert faults == []

    def test_all_up_no_fault(self) -> None:
        store = make_fake_metric_store()
        inject_nic_up(store, node_id="node-0", device="ib0")
        inject_nic_up(store, node_id="node-0", device="ib0")

        df = _build_nic_df(store)
        faults = _analyze_nic_persistent_down(df)

        assert faults == []

    def test_multiple_devices_one_persistent(self) -> None:
        """Two devices on same node: ib0 persistent down, ib1 recovered."""
        store = make_fake_metric_store()
        inject_nic_up(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib0")
        inject_nic_up(store, node_id="node-0", device="ib1")
        inject_nic_down(store, node_id="node-0", device="ib1")
        inject_nic_up(store, node_id="node-0", device="ib1")

        df = _build_nic_df(store)
        faults = _analyze_nic_persistent_down(df)

        assert len(faults) == 1
        assert "ib0" in faults[0].reason
        assert "ib1" not in faults[0].reason

    def test_multiple_devices_both_persistent_aggregated(self) -> None:
        """Two devices on same node both persistently down: single fault listing both."""
        store = make_fake_metric_store()
        inject_nic_up(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib0")
        inject_nic_up(store, node_id="node-0", device="ib1")
        inject_nic_down(store, node_id="node-0", device="ib1")

        df = _build_nic_df(store)
        faults = _analyze_nic_persistent_down(df)

        assert len(faults) == 1
        assert "ib0" in faults[0].reason
        assert "ib1" in faults[0].reason

    def test_multiple_nodes_independent(self) -> None:
        store = make_fake_metric_store()
        # node-0: persistent down
        inject_nic_up(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib0")
        # node-1: recovered
        inject_nic_up(store, node_id="node-1", device="ib0")
        inject_nic_down(store, node_id="node-1", device="ib0")
        inject_nic_up(store, node_id="node-1", device="ib0")

        df = _build_nic_df(store)
        faults = _analyze_nic_persistent_down(df)

        assert len(faults) == 1
        assert faults[0].node_id == "node-0"

    def test_reason_lists_device_names_sorted(self) -> None:
        store = make_fake_metric_store()
        inject_nic_up(store, node_id="node-0", device="ib2")
        inject_nic_down(store, node_id="node-0", device="ib2")
        inject_nic_up(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib0")

        df = _build_nic_df(store)
        faults = _analyze_nic_persistent_down(df)

        assert len(faults) == 1
        assert "ib0, ib2" in faults[0].reason


# ---------------------------------------------------------------------------
# check_all_nic_faults (combined: flap + persistent)
# ---------------------------------------------------------------------------


class TestCheckAllNicFaults:
    def test_empty_store_returns_empty(self) -> None:
        store = make_fake_metric_store()
        faults = check_all_nic_faults(store, window=WINDOW, flap_threshold=2)
        assert faults == []

    def test_flap_only_detected(self) -> None:
        """NIC flapping that ends up: flap fires but not persistent."""
        store = make_fake_metric_store()
        inject_nic_up(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib0")
        inject_nic_up(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib0")
        inject_nic_up(store, node_id="node-0", device="ib0")

        faults = check_all_nic_faults(store, window=WINDOW, flap_threshold=2)

        assert len(faults) == 1
        assert "went down" in faults[0].reason

    def test_persistent_only_detected(self) -> None:
        """Single up→down (below flap threshold) but persistent: detected."""
        store = make_fake_metric_store()
        inject_nic_up(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib0")

        faults = check_all_nic_faults(store, window=WINDOW, flap_threshold=5)

        assert len(faults) == 1
        assert "persistently down" in faults[0].reason

    def test_both_flap_and_persistent_deduped_by_node(self) -> None:
        """Same node triggers both flap and persistent: persistent fault wins."""
        store = make_fake_metric_store()
        inject_nic_up(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib0")
        inject_nic_up(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib0")

        faults = check_all_nic_faults(store, window=WINDOW, flap_threshold=2)

        assert len(faults) == 1
        assert faults[0].node_id == "node-0"
        assert "persistently down" in faults[0].reason

    def test_different_nodes_different_fault_types(self) -> None:
        """node-0: flap only (ends up), node-1: persistent only (below threshold)."""
        store = make_fake_metric_store()
        # node-0: 2 transitions ending up → flap at threshold=2, not persistent
        inject_nic_up(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib0")
        inject_nic_up(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib0")
        inject_nic_up(store, node_id="node-0", device="ib0")
        # node-1: 1 transition ending down → persistent, below flap threshold
        inject_nic_up(store, node_id="node-1", device="ib0")
        inject_nic_down(store, node_id="node-1", device="ib0")

        faults = check_all_nic_faults(store, window=WINDOW, flap_threshold=2)

        assert len(faults) == 2
        node_ids = {f.node_id for f in faults}
        assert node_ids == {"node-0", "node-1"}

    def test_no_fault_when_healthy(self) -> None:
        store = make_fake_metric_store()
        inject_nic_up(store, node_id="node-0", device="ib0")
        inject_nic_up(store, node_id="node-0", device="ib0")
        inject_nic_up(store, node_id="node-0", device="ib0")

        faults = check_all_nic_faults(store, window=WINDOW, flap_threshold=1)

        assert faults == []

    def test_returns_non_ephemeral_faults(self) -> None:
        store = make_fake_metric_store()
        inject_nic_up(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib0")
        inject_nic_up(store, node_id="node-0", device="ib0")
        inject_nic_down(store, node_id="node-0", device="ib0")

        faults = check_all_nic_faults(store, window=WINDOW, flap_threshold=2)

        assert len(faults) == 1
        assert faults[0].ephemeral is False


# ---------------------------------------------------------------------------
# check_majority_nic_down
# ---------------------------------------------------------------------------


class TestCheckMajorityNicDown:
    def test_exactly_half_nics_down_does_not_trigger(self) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(
            target_id="node-0",
            samples=[
                GaugeSample(name=NODE_NETWORK_UP, labels={"interface": "eth0"}, value=1.0),
                GaugeSample(name=NODE_NETWORK_UP, labels={"interface": "eth1"}, value=0.0),
            ],
        )

        result = check_majority_nic_down(store)
        assert result == []

    def test_majority_nics_down_triggers(self) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(
            target_id="node-0",
            samples=[
                GaugeSample(name=NODE_NETWORK_UP, labels={"interface": "eth0"}, value=0.0),
                GaugeSample(name=NODE_NETWORK_UP, labels={"interface": "eth1"}, value=0.0),
                GaugeSample(name=NODE_NETWORK_UP, labels={"interface": "eth2"}, value=1.0),
            ],
        )

        result = check_majority_nic_down(store)
        assert len(result) == 1
        assert result[0].node_id == "node-0"
        assert result[0].ephemeral is False

    def test_all_nics_up_returns_empty(self) -> None:
        store = make_fake_metric_store()
        store.ingest_samples(
            target_id="node-0",
            samples=[
                GaugeSample(name=NODE_NETWORK_UP, labels={"interface": "eth0"}, value=1.0),
                GaugeSample(name=NODE_NETWORK_UP, labels={"interface": "eth1"}, value=1.0),
            ],
        )

        result = check_majority_nic_down(store)
        assert result == []

    def test_empty_metric_store_returns_empty(self) -> None:
        store = make_fake_metric_store()
        result = check_majority_nic_down(store)
        assert result == []


# ---------------------------------------------------------------------------
# check_disk_fault
# ---------------------------------------------------------------------------


class TestCheckDiskFault:
    def test_below_threshold_returns_fault(self) -> None:
        store = make_fake_metric_store()
        inject_disk_fault(store, node_id="node-0", mountpoint="/", available_bytes=500e6)

        result = check_disk_fault(store)

        assert len(result) == 1
        assert result[0].node_id == "node-0"
        assert "disk space low" in result[0].reason

    def test_above_threshold_returns_empty(self) -> None:
        store = make_fake_metric_store()
        inject_disk_fault(store, node_id="node-0", mountpoint="/", available_bytes=100e9)

        assert check_disk_fault(store) == []

    def test_empty_store_returns_empty(self) -> None:
        store = make_fake_metric_store()

        assert check_disk_fault(store) == []

    def test_multiple_nodes_only_low_ones_flagged(self) -> None:
        store = make_fake_metric_store()
        inject_disk_fault(store, node_id="node-0", mountpoint="/", available_bytes=200e6)
        inject_disk_fault(store, node_id="node-1", mountpoint="/", available_bytes=50e9)

        result = check_disk_fault(store)

        assert len(result) == 1
        assert result[0].node_id == "node-0"

    def test_custom_threshold(self) -> None:
        store = make_fake_metric_store()
        inject_disk_fault(store, node_id="node-0", mountpoint="/", available_bytes=5e9)

        result_default = check_disk_fault(store)
        assert result_default == []

        result_high = check_disk_fault(store, disk_available_threshold_bytes=10e9)
        assert len(result_high) == 1
        assert result_high[0].node_id == "node-0"

    def test_exactly_at_threshold_does_not_trigger(self) -> None:
        store = make_fake_metric_store()
        inject_disk_fault(store, node_id="node-0", mountpoint="/", available_bytes=1e9)

        result = check_disk_fault(store, disk_available_threshold_bytes=1e9)
        assert result == []
