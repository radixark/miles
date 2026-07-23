from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

from miles.utils.weight_target import pin_and_verify


class FakeTarget:
    """Records calls; ``versions`` gives the version reported by read_version()
    on each successive call (repeats the last entry once exhausted)."""

    def __init__(self, versions, *, fail_loads: int = 0, hang: bool = False):
        self.versions = list(versions)
        self.fail_loads = fail_loads
        self.hang = hang
        self.loads: list[tuple[str, str]] = []
        self._reads = 0

    async def load_from_disk(self, hf_dir, weight_version):
        if self.hang:
            import asyncio

            await asyncio.sleep(10)
        if self.fail_loads > 0:
            self.fail_loads -= 1
            raise RuntimeError("transient load failure")
        self.loads.append((hf_dir, weight_version))

    async def read_version(self):
        idx = min(self._reads, len(self.versions) - 1)
        self._reads += 1
        return self.versions[idx]


async def test_pin_and_verify_succeeds_first_try():
    target = FakeTarget(versions=["5"])
    ok = await pin_and_verify([target], "/snap", "5")
    assert ok
    assert target.loads == [("/snap", "5")]


async def test_pin_and_verify_all_targets_must_match():
    a, b = FakeTarget(versions=["5"]), FakeTarget(versions=["4"])
    ok = await pin_and_verify([a, b], "/snap", "5", retries=1)
    assert not ok


async def test_pin_and_verify_retries_on_version_mismatch_then_succeeds():
    # First read reports stale "4", second read (after a re-load) reports "5".
    target = FakeTarget(versions=["4", "5"])
    ok = await pin_and_verify([target], "/snap", "5", retries=2)
    assert ok
    assert len(target.loads) == 2  # one load per attempt


async def test_pin_and_verify_exhausts_retries_and_returns_false():
    target = FakeTarget(versions=["999"])
    ok = await pin_and_verify([target], "/snap", "5", retries=2)
    assert not ok
    assert len(target.loads) == 2


async def test_pin_and_verify_transient_load_error_is_retried():
    target = FakeTarget(versions=["5"], fail_loads=1)
    ok = await pin_and_verify([target], "/snap", "5", retries=2)
    assert ok
    assert target.loads == [("/snap", "5")]  # the failed attempt logged nothing


async def test_pin_and_verify_never_raises_on_timeout():
    target = FakeTarget(versions=["5"], hang=True)
    ok = await pin_and_verify([target], "/snap", "5", timeout=0.05, retries=1)
    assert not ok


async def test_pin_and_verify_empty_targets_is_never_pinned():
    # No engines to confirm against: cannot claim the pin succeeded.
    ok = await pin_and_verify([], "/snap", "5")
    assert not ok
