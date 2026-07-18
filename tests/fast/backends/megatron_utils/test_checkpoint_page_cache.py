import os
from pathlib import Path

from miles.backends.megatron_utils import checkpoint


def test_drop_checkpoint_page_cache_advises_every_file(tmp_path: Path, monkeypatch):
    (tmp_path / "release").mkdir()
    first = tmp_path / "latest_checkpointed_iteration.txt"
    second = tmp_path / "release" / "__0_0.distcp"
    first.write_bytes(b"release")
    second.write_bytes(b"checkpoint")
    advised = []

    def record_fadvise(fd, offset, length, advice):
        advised.append((Path(os.readlink(f"/proc/self/fd/{fd}")).name, offset, length, advice))

    monkeypatch.setattr(checkpoint.os, "posix_fadvise", record_fadvise)

    file_count, total_bytes = checkpoint._drop_checkpoint_page_cache(tmp_path)

    assert file_count == 2
    assert total_bytes == len(b"release") + len(b"checkpoint")
    assert sorted(name for name, _, _, _ in advised) == ["__0_0.distcp", "latest_checkpointed_iteration.txt"]
    assert all(offset == 0 and length == 0 for _, offset, length, _ in advised)
    assert all(advice == os.POSIX_FADV_DONTNEED for _, _, _, advice in advised)
