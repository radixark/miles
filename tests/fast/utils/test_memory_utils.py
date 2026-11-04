from miles.utils import memory_utils


def test_clear_memory_can_skip_the_full_garbage_collection(monkeypatch) -> None:
    """A full gc.collect() costs about a second in a trainer process; a caller that just ran one can skip it."""
    calls = []
    monkeypatch.setattr(memory_utils.torch.accelerator, "synchronize", lambda: calls.append("synchronize"))
    monkeypatch.setattr(memory_utils.torch.accelerator, "empty_cache", lambda: calls.append("empty_cache"))
    monkeypatch.setattr(memory_utils.gc, "collect", lambda: calls.append("collect") or 0)

    memory_utils.clear_memory()
    memory_utils.clear_memory(collect_garbage=False)

    assert calls == ["synchronize", "collect", "empty_cache", "synchronize", "empty_cache"]
