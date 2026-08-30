from unittest.mock import Mock

from miles.utils import metric_utils


def test_has_repetition_checks_stride_aligned_middle_window(monkeypatch) -> None:
    text = "a" * 5_000 + "b" * 10_000 + "c" * 5_000
    compression_ratio = Mock(
        side_effect=lambda window: (11.0, 0.0) if window == "b" * 10_000 else (1.0, 0.0)
    )
    monkeypatch.setattr(metric_utils, "compression_ratio", compression_ratio)

    assert metric_utils.has_repetition(text)
    assert compression_ratio.call_count == 2


def test_has_repetition_checks_unaligned_final_window(monkeypatch) -> None:
    text = "a" * 2_345 + "b" * 10_000
    final_window = text[-10_000:]
    compression_ratio = Mock(
        side_effect=lambda window: (11.0, 0.0) if window == final_window else (1.0, 0.0)
    )
    monkeypatch.setattr(metric_utils, "compression_ratio", compression_ratio)

    assert metric_utils.has_repetition(text)


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
