import hashlib
from unittest.mock import Mock

from miles.utils import metric_utils


def test_has_repetition_checks_stride_aligned_middle_window(monkeypatch) -> None:
    text = "a" * 5_000 + "b" * 10_000 + "c" * 5_000
    compression_ratio = Mock(side_effect=lambda window: (11.0, 0.0) if window == "b" * 10_000 else (1.0, 0.0))
    monkeypatch.setattr(metric_utils, "compression_ratio", compression_ratio)

    assert metric_utils.has_repetition(text)
    assert compression_ratio.call_count == 2


def test_has_repetition_checks_unaligned_final_window(monkeypatch) -> None:
    text = "a" * 2_345 + "b" * 10_000
    final_window = text[-10_000:]
    compression_ratio = Mock(side_effect=lambda window: (11.0, 0.0) if window == final_window else (1.0, 0.0))
    monkeypatch.setattr(metric_utils, "compression_ratio", compression_ratio)

    assert metric_utils.has_repetition(text)
    assert compression_ratio.call_count == 2


def test_has_repetition_checks_exactly_ten_thousand_characters() -> None:
    assert metric_utils.has_repetition("repeat" * 1_666 + "xxxx")


def test_has_repetition_ignores_short_text(monkeypatch) -> None:
    compression_ratio = Mock(return_value=(100.0, 0.0))
    monkeypatch.setattr(metric_utils, "compression_ratio", compression_ratio)

    assert not metric_utils.has_repetition("x" * 9_999)
    compression_ratio.assert_not_called()


def test_has_repetition_uses_strict_threshold(monkeypatch) -> None:
    monkeypatch.setattr(metric_utils, "compression_ratio", lambda _: (10.0, 0.0))

    assert not metric_utils.has_repetition("x" * 10_000)


def test_repetition_window_count_is_bounded_for_very_long_text(monkeypatch) -> None:
    """The scan runs on the rollout executor's event loop; its cost must not
    scale with response length."""
    compression_ratio = Mock(return_value=(1.0, 0.0))
    monkeypatch.setattr(metric_utils, "compression_ratio", compression_ratio)

    assert not metric_utils.has_repetition("x" * 2_000_000)  # ~400 natural stride windows
    assert compression_ratio.call_count <= metric_utils.REPETITION_MAX_WINDOWS


def test_bounded_windows_keep_first_and_final_coverage() -> None:
    text = "".join(chr(33 + (i % 90)) for i in range(500_000))
    windows = list(metric_utils._repetition_windows(text))

    assert len(windows) <= metric_utils.REPETITION_MAX_WINDOWS
    assert windows[0] == text[: metric_utils.REPETITION_WINDOW_SIZE_CHARS]
    assert windows[-1] == text[-metric_utils.REPETITION_WINDOW_SIZE_CHARS :]


def test_repetition_mid_way_in_a_very_long_response_is_still_caught() -> None:
    """Up to ~320k chars the widened stride keeps windows overlapping, so a
    repetitive run longer than one window always lands fully inside some
    window (real compression, no mocks)."""
    filler = "".join(hashlib.sha256(str(i).encode()).hexdigest() for i in range(4_300))  # ~275k, low ratio
    text = filler[:150_000] + "spam" * 6_250 + filler[150_000:]

    assert metric_utils.has_repetition(text)
    assert not metric_utils.has_repetition(filler)
