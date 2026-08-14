import concurrent.futures
import gc
import logging
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp
from tests.fast.dist_utils import find_free_port, init_gloo
from torch.distributed import HashStore, Store, TCPStore

from miles.utils.distributed_lock import StoreTicketLock, create_world_ticket_lock
from miles.utils.distributed_utils import init_gloo_group

PREFIX = "test/lock"
NEXT_KEY = f"{PREFIX}/next"
SERVING_KEY = f"{PREFIX}/serving"
_LOCK_MODULE = "miles.utils.distributed_lock"


class _BrokenStore:
    def add(self, key: str, amount: int) -> int:
        raise RuntimeError("store is gone")


class _ScriptedStore:
    def __init__(self, *, ticket: int, serve_after_polls: int, prefix: str = PREFIX) -> None:
        self._ticket = ticket
        self._serve_after_polls = serve_after_polls
        self._next_key = f"{prefix}/next"
        self._serving_key = f"{prefix}/serving"
        self.polls = 0

    def add(self, key: str, amount: int) -> int:
        if key == self._next_key:
            return self._ticket + 1
        assert key == self._serving_key, f"unexpected key {key}"
        self.polls += 1
        return self._ticket if self.polls > self._serve_after_polls else 0


class _UnservableStore:
    def add(self, key: str, amount: int) -> int:
        if key == SERVING_KEY and amount != 0:
            raise RuntimeError("store is gone")
        return 0 if amount == 0 else 1


class _FakeClock:
    def __init__(self, *, step: float, start: float = 0.0) -> None:
        self.now = start
        self._step = step
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += self._step


def _wait_until(predicate: Callable[[], bool], *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out waiting for the condition"
        time.sleep(0.005)


def _wait_until_queued(store: Store, *, drawn: int) -> None:
    _wait_until(lambda: store.add(NEXT_KEY, 0) == drawn)


def _grant_everyone(store: Store) -> None:
    while store.add(SERVING_KEY, 0) < store.add(NEXT_KEY, 0):
        store.add(SERVING_KEY, 1)


@contextmanager
def _pool(store: Store | None = None, *, max_workers: int = 1) -> Iterator[ThreadPoolExecutor]:
    pool = ThreadPoolExecutor(max_workers=max_workers)
    try:
        yield pool
    finally:
        if store is not None:
            try:
                _grant_everyone(store)
            except RuntimeError:
                pass
        pool.shutdown(wait=False, cancel_futures=True)


class TestStoreTicketLock:
    def test_the_first_acquirer_takes_ticket_zero_without_waiting(self) -> None:
        """Nobody is queued, so acquiring draws ticket 0 and returns at once."""
        store = HashStore()
        lock = StoreTicketLock(store=store, prefix=PREFIX)

        lock.acquire()

        assert store.add(NEXT_KEY, 0) == 1
        assert store.add(SERVING_KEY, 0) == 0

    def test_release_calls_the_next_ticket(self) -> None:
        """Releasing advances the serving counter so the next ticket holder may proceed."""
        store = HashStore()

        with StoreTicketLock(store=store, prefix=PREFIX):
            pass

        assert store.add(SERVING_KEY, 0) == 1

    def test_enter_returns_the_lock_instance(self) -> None:
        """``with lock as l`` must bind the lock itself, not None."""
        lock = StoreTicketLock(store=HashStore(), prefix=PREFIX)

        with lock as entered:
            assert entered is lock

    def test_a_released_lock_can_be_reacquired_with_a_fresh_ticket(self) -> None:
        """Release resets the held state, so the same object can queue again."""
        store = HashStore()
        lock = StoreTicketLock(store=store, prefix=PREFIX)
        lock.acquire()
        lock.release()

        lock.acquire()

        assert store.add(NEXT_KEY, 0) == 2
        assert store.add(SERVING_KEY, 0) == 1
        lock.release()
        assert store.add(SERVING_KEY, 0) == 2

    def test_a_second_release_raises_instead_of_calling_a_strangers_ticket(self) -> None:
        """Release drops the held ticket, so a double release must fail loudly."""
        store = HashStore()
        lock = StoreTicketLock(store=store, prefix=PREFIX)
        lock.acquire()
        lock.release()

        with pytest.raises(AssertionError):
            lock.release()

        assert store.add(SERVING_KEY, 0) == 1

    def test_a_raising_critical_section_keeps_the_lock(self) -> None:
        """Failing closed: the failed holder's remote work may outlive the exception."""
        store = HashStore()
        lock = StoreTicketLock(store=store, prefix=PREFIX)

        with pytest.raises(ValueError):
            with lock:
                raise ValueError("bucket failed")

        assert store.add(SERVING_KEY, 0) == 0
        lock.release()
        assert store.add(SERVING_KEY, 0) == 1

    def test_a_raising_base_exception_also_keeps_the_lock(self) -> None:
        """Fail closed on any BaseException, not just Exception subclasses."""
        store = HashStore()
        lock = StoreTicketLock(store=store, prefix=PREFIX)

        with pytest.raises(KeyboardInterrupt):
            with lock:
                raise KeyboardInterrupt

        assert store.add(SERVING_KEY, 0) == 0

    def test_releasing_an_unheld_lock_raises(self) -> None:
        """A release without a matching acquire would hand the lock to a stranger."""
        store = HashStore()
        lock = StoreTicketLock(store=store, prefix=PREFIX)

        with pytest.raises(AssertionError):
            lock.release()

        assert store.add(SERVING_KEY, 0) == 0

    def test_acquiring_twice_raises_instead_of_deadlocking(self) -> None:
        """The lock is not reentrant, and says so rather than waiting on its own ticket."""
        store = HashStore()
        lock = StoreTicketLock(store=store, prefix=PREFIX)
        lock.acquire()

        with pytest.raises(AssertionError):
            lock.acquire()

        assert store.add(NEXT_KEY, 0) == 1

    def test_acquire_blocks_until_the_previous_holder_releases(self) -> None:
        """A drawn ticket only becomes the lock once serving catches up with it."""
        store = HashStore()
        store.add(NEXT_KEY, 1)
        lock = StoreTicketLock(store=store, prefix=PREFIX, poll_interval=0.001)

        with _pool(store) as pool:
            waiter = pool.submit(lock.acquire)
            _wait_until_queued(store, drawn=2)
            with pytest.raises(concurrent.futures.TimeoutError):
                waiter.result(timeout=0.2)

            store.add(SERVING_KEY, 1)
            waiter.result(timeout=5.0)

    def test_a_holder_that_never_releases_stalls_the_queue(self) -> None:
        """There is no lease: an abandoned ticket blocks the queue rather than timing out."""
        store = HashStore()
        StoreTicketLock(store=store, prefix=PREFIX).acquire()
        waiter_lock = StoreTicketLock(store=store, prefix=PREFIX, poll_interval=0.001)

        with _pool(store) as pool:
            waiter = pool.submit(waiter_lock.acquire)
            _wait_until_queued(store, drawn=2)
            with pytest.raises(concurrent.futures.TimeoutError):
                waiter.result(timeout=0.3)

            store.add(SERVING_KEY, 1)
            waiter.result(timeout=5.0)

    def test_locks_on_different_prefixes_do_not_block_each_other(self) -> None:
        """One store carries many independent queues, so prefixes must not share counters."""
        store = HashStore()
        held = StoreTicketLock(store=store, prefix=PREFIX)
        held.acquire()

        with StoreTicketLock(store=store, prefix="test/other", poll_interval=0.001):
            pass

        assert store.add(SERVING_KEY, 0) == 0
        assert store.add("test/other/serving", 0) == 1

    def test_a_long_wait_is_logged_periodically(self, caplog) -> None:
        """A stuck queue reports which ticket it holds and which one is being served."""
        store = HashStore()
        store.add(NEXT_KEY, 1)
        lock = StoreTicketLock(store=store, prefix=PREFIX, poll_interval=0.001, warn_interval=0.0)

        with caplog.at_level(logging.WARNING, logger=_LOCK_MODULE):
            with _pool(store) as pool:
                waiter = pool.submit(lock.acquire)
                _wait_until(lambda: len(caplog.records) >= 2)
                store.add(SERVING_KEY, 1)
                waiter.result(timeout=5.0)

        assert "holding ticket 1, now serving 0" in caplog.records[0].message

    def test_wait_logging_is_throttled_to_the_warn_interval(self, caplog) -> None:
        """Without throttling a long wait would log once per poll, which is once every 10ms."""
        store = _ScriptedStore(ticket=4, serve_after_polls=13)
        lock = StoreTicketLock(store=store, prefix=PREFIX, warn_interval=60.0)

        with caplog.at_level(logging.WARNING, logger=_LOCK_MODULE):
            with patch(f"{_LOCK_MODULE}.time", _FakeClock(step=10.0)):
                lock.acquire()

        assert store.polls == 14
        assert len(caplog.records) == 2
        assert "holding ticket 4, now serving 0" in caplog.records[0].message

    def test_no_warning_fires_before_a_full_warn_interval_has_passed(self, caplog) -> None:
        """The first warning is measured from acquire time, not from clock zero."""
        store = _ScriptedStore(ticket=4, serve_after_polls=5)
        lock = StoreTicketLock(store=store, prefix=PREFIX, warn_interval=60.0)

        with caplog.at_level(logging.WARNING, logger=_LOCK_MODULE):
            with patch(f"{_LOCK_MODULE}.time", _FakeClock(step=10.0, start=1_000_000.0)):
                lock.acquire()

        assert caplog.records == []

    def test_acquire_sleeps_the_configured_poll_interval_between_polls(self) -> None:
        """Each failed poll sleeps exactly the configured interval instead of busy-spinning."""
        store = _ScriptedStore(ticket=4, serve_after_polls=3)
        lock = StoreTicketLock(store=store, prefix=PREFIX, poll_interval=0.25)
        clock = _FakeClock(step=0.0)

        with patch(f"{_LOCK_MODULE}.time", clock):
            lock.acquire()

        assert clock.sleeps == [0.25, 0.25, 0.25]

    def test_a_dead_store_surfaces_as_an_error_rather_than_a_hang(self) -> None:
        """Fail-fast is the whole crash story, so store errors must propagate out of acquire."""
        lock = StoreTicketLock(store=_BrokenStore(), prefix=PREFIX)

        with pytest.raises(RuntimeError):
            lock.acquire()

    def test_release_propagates_a_store_failure(self) -> None:
        """An unlock that never reached the store must raise instead of reporting success."""
        lock = StoreTicketLock(store=_UnservableStore(), prefix=PREFIX)
        lock.acquire()

        with pytest.raises(RuntimeError):
            lock.release()


class TestStoreTicketLockContention:
    def test_the_lock_is_granted_in_ticket_order(self) -> None:
        """Ticket order is the grant order, which is what makes the queue starvation-free."""
        store = HashStore()
        holder = StoreTicketLock(store=store, prefix=PREFIX, poll_interval=0.001)
        holder.acquire()

        granted: list[int] = []

        def contend(index: int) -> None:
            with StoreTicketLock(store=store, prefix=PREFIX, poll_interval=0.001):
                granted.append(index)

        with _pool(store, max_workers=3) as pool:
            waiters = []
            for index in range(3):
                waiters.append(pool.submit(contend, index))
                _wait_until_queued(store, drawn=index + 2)

            holder.release()
            for waiter in waiters:
                waiter.result(timeout=5.0)

        assert granted == [0, 1, 2]

    def test_the_critical_section_runs_alone(self) -> None:
        """A deliberately non-atomic read-modify-write loses no update under the lock."""
        store = HashStore()
        counter = [0]
        contenders = 4
        iterations = 5
        starting_line = threading.Barrier(contenders, timeout=30.0)

        def contend() -> None:
            for _ in range(iterations):
                starting_line.wait()
                with StoreTicketLock(store=store, prefix=PREFIX, poll_interval=0.001):
                    value = counter[0]
                    time.sleep(0.002)
                    counter[0] = value + 1

        with _pool(store, max_workers=contenders) as pool:
            waiters = [pool.submit(contend) for _ in range(contenders)]
            for waiter in waiters:
                waiter.result(timeout=30.0)

        assert counter[0] == contenders * iterations


def _host_worker(ready_path: str) -> None:
    store = TCPStore(host_name="0.0.0.0", port=0, is_master=True, wait_for_workers=False)
    Path(ready_path).write_text(str(store.port))
    time.sleep(60.0)


@contextmanager
def _store_host(ready_path: Path) -> Iterator[int]:
    process = mp.get_context("spawn").Process(target=_host_worker, args=(str(ready_path),))
    process.start()
    try:
        _wait_until(lambda: ready_path.exists() and ready_path.read_text() != "", timeout=60.0)
        yield int(ready_path.read_text())
    finally:
        process.terminate()
        process.join(timeout=30.0)
        assert not process.is_alive(), "the store host survived terminate()"


class TestStoreTicketLockOverTcpStore:
    def test_separate_connections_to_one_store_form_a_single_queue(self) -> None:
        """Every rank holds its own connection, so the queue must span connections, not objects."""
        host = TCPStore(host_name="0.0.0.0", port=0, is_master=True, wait_for_workers=False)
        holder = StoreTicketLock(store=host, prefix=PREFIX)
        holder.acquire()

        client = TCPStore(host_name="127.0.0.1", port=host.port, is_master=False)
        waiter_lock = StoreTicketLock(store=client, prefix=PREFIX, poll_interval=0.001)

        with _pool(host) as pool:
            waiter = pool.submit(waiter_lock.acquire)
            _wait_until_queued(host, drawn=2)
            with pytest.raises(concurrent.futures.TimeoutError):
                waiter.result(timeout=0.3)

            holder.release()
            waiter.result(timeout=10.0)
            assert client.add(SERVING_KEY, 0) == 1

    def test_a_holder_connection_that_disappears_keeps_holding_its_ticket(self) -> None:
        """A lost connection is not a release: a dead holder is cleaned up by teardown, not by the lock."""
        host = TCPStore(host_name="0.0.0.0", port=0, is_master=True, wait_for_workers=False)
        dying = StoreTicketLock(store=TCPStore(host_name="127.0.0.1", port=host.port, is_master=False), prefix=PREFIX)
        dying.acquire()

        del dying
        gc.collect()

        waiter_lock = StoreTicketLock(store=host, prefix=PREFIX, poll_interval=0.001)
        with _pool(host) as pool:
            waiter = pool.submit(waiter_lock.acquire)
            _wait_until_queued(host, drawn=2)
            with pytest.raises(concurrent.futures.TimeoutError):
                waiter.result(timeout=0.3)

            host.add(SERVING_KEY, 1)
            waiter.result(timeout=10.0)

    def test_a_waiter_fails_fast_when_the_host_dies_while_it_polls(self, tmp_path) -> None:
        """The documented crash path: a queued rank must raise, not poll a store nobody serves."""
        with _pool() as pool:
            with _store_host(tmp_path / "port") as port:
                holder_store = TCPStore(host_name="127.0.0.1", port=port, is_master=False)
                StoreTicketLock(store=holder_store, prefix=PREFIX).acquire()
                waiter_lock = StoreTicketLock(
                    store=TCPStore(host_name="127.0.0.1", port=port, is_master=False),
                    prefix=PREFIX,
                    poll_interval=0.01,
                )

                waiter = pool.submit(waiter_lock.acquire)
                _wait_until_queued(holder_store, drawn=2)
                with pytest.raises(concurrent.futures.TimeoutError):
                    waiter.result(timeout=0.3)

            with pytest.raises(RuntimeError):
                waiter.result(timeout=30.0)

    def test_acquire_fails_fast_once_the_host_process_is_gone(self, tmp_path) -> None:
        """Losing the store's host must raise instead of spinning on a ticket forever."""
        with _store_host(tmp_path / "port") as port:
            lock = StoreTicketLock(store=TCPStore(host_name="127.0.0.1", port=port, is_master=False), prefix=PREFIX)
            lock.acquire()
            lock.release()

        with _pool() as pool:
            failing = pool.submit(lock.acquire)
            with pytest.raises(RuntimeError):
                failing.result(timeout=30.0)


def _spawn_ranks(worker: Callable, args: tuple, nprocs: int, *, timeout: float = 180.0) -> None:
    context = mp.spawn(worker, args=args, nprocs=nprocs, join=False)
    deadline = time.monotonic() + timeout
    try:
        while not context.join(timeout=1.0):
            if time.monotonic() >= deadline:
                pytest.fail(f"ranks did not finish within {timeout}s; a collective is likely asymmetric")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=30.0)


def _increment_under_lock(
    lock: AbstractContextManager, counter_path: str, iterations: int, *, barrier: Callable[[], None] | None = None
) -> None:
    for _ in range(iterations):
        if barrier is not None:
            barrier()
        with lock:
            value = int(Path(counter_path).read_text())
            time.sleep(0.01)
            Path(counter_path).write_text(str(value + 1))


def _world_lock_worker(rank: int, world_size: int, port: int, counter_path: str, iterations: int) -> None:
    init_gloo(rank, world_size, port=port)
    init_gloo_group()

    lock = create_world_ticket_lock(prefix=PREFIX)
    _increment_under_lock(lock, counter_path, iterations, barrier=dist.barrier)

    dist.barrier()
    dist.destroy_process_group()


def _world_lock_abstainer_worker(
    rank: int, world_size: int, port: int, counter_dir: str, iterations: int, abstainers_per_round: list[list[int]]
) -> None:
    init_gloo(rank, world_size, port=port)
    init_gloo_group()

    for round_index, abstainers in enumerate(abstainers_per_round):
        participates = rank not in abstainers
        lock = create_world_ticket_lock(prefix=f"{PREFIX}/round{round_index}", participates=participates)
        assert isinstance(lock, StoreTicketLock) == participates
        if participates:
            _increment_under_lock(lock, f"{counter_dir}/{round_index}", iterations)
        else:
            with lock:
                pass
        dist.barrier()

    dist.destroy_process_group()


class TestCreateWorldTicketLock:
    def test_every_rank_shares_the_store_hosted_by_rank_zero(self, tmp_path) -> None:
        """The address broadcast makes one lock out of the whole world, not one per rank."""
        counter_path = tmp_path / "counter"
        counter_path.write_text("0")
        world_size = 3
        iterations = 3

        _spawn_ranks(_world_lock_worker, (world_size, find_free_port(), str(counter_path), iterations), world_size)

        assert int(counter_path.read_text()) == world_size * iterations

    def test_a_rank_that_does_not_contend_joins_the_broadcast_but_holds_no_lock(self, tmp_path) -> None:
        """Abstainers still enter the collective, whether the abstainer is the host, a client, or everyone."""
        world_size = 3
        iterations = 3
        abstainers_per_round = [[1], [0], [0, 1, 2]]
        for round_index in range(len(abstainers_per_round)):
            (tmp_path / str(round_index)).write_text("0")

        _spawn_ranks(
            _world_lock_abstainer_worker,
            (world_size, find_free_port(), str(tmp_path), iterations, abstainers_per_round),
            world_size,
        )

        for round_index, abstainers in enumerate(abstainers_per_round):
            contenders = world_size - len(abstainers)
            assert int((tmp_path / str(round_index)).read_text()) == contenders * iterations


class TestCreateWorldTicketLockAddressing:
    @staticmethod
    @contextmanager
    def _world(*, rank: int, broadcast_address: str | None) -> Iterator[MagicMock]:
        def broadcast(object_list, src, group) -> None:
            if broadcast_address is not None:
                object_list[0] = broadcast_address

        dist_mock = MagicMock()
        dist_mock.get_rank.return_value = rank
        dist_mock.broadcast_object_list.side_effect = broadcast
        with (
            patch(f"{_LOCK_MODULE}.dist", dist_mock),
            patch(f"{_LOCK_MODULE}.get_gloo_group") as get_gloo_group,
            patch(f"{_LOCK_MODULE}.get_current_node_ip", return_value="10.1.2.3"),
            patch(f"{_LOCK_MODULE}.TCPStore") as store_class,
        ):
            store_class.return_value.port = 4567
            yield SimpleNamespace(store_class=store_class, dist=dist_mock, gloo_group=get_gloo_group.return_value)

    def test_the_host_publishes_the_node_address_of_its_wildcard_bound_store(self) -> None:
        """Ranks on other nodes cannot reach 0.0.0.0, so the routable node IP is what goes out."""
        with self._world(rank=0, broadcast_address=None) as mocks:
            create_world_ticket_lock(prefix=PREFIX)

        mocks.store_class.assert_called_once_with(host_name="0.0.0.0", port=0, is_master=True, wait_for_workers=False)
        published = mocks.dist.broadcast_object_list.call_args.args[0]
        assert published == ["10.1.2.3:4567"]

    def test_a_client_connects_to_the_address_it_was_given(self) -> None:
        """A client that assumed localhost would silently connect to its own node instead."""
        with self._world(rank=1, broadcast_address="10.9.9.9:4567") as mocks:
            create_world_ticket_lock(prefix=PREFIX)

        mocks.store_class.assert_called_once_with(host_name="10.9.9.9", port=4567, is_master=False)

    def test_a_client_connects_to_an_ipv6_address_it_was_given(self) -> None:
        """Only the rightmost colon separates the port, so an IPv6 host must survive intact."""
        with self._world(rank=1, broadcast_address="2001:db8::9:4567") as mocks:
            create_world_ticket_lock(prefix=PREFIX)

        mocks.store_class.assert_called_once_with(host_name="2001:db8::9", port=4567, is_master=False)

    def test_the_address_broadcast_runs_on_the_dedicated_gloo_group(self) -> None:
        """Broadcasting python objects over the default group would break on NCCL worlds."""
        with self._world(rank=0, broadcast_address=None) as mocks:
            create_world_ticket_lock(prefix=PREFIX)

        call = mocks.dist.broadcast_object_list.call_args
        assert call.kwargs["src"] == 0
        assert call.kwargs["group"] is mocks.gloo_group

    def test_the_factory_forwards_the_store_prefix_and_poll_interval(self) -> None:
        """The returned lock must queue on the shared store under the caller's prefix and cadence."""
        with self._world(rank=1, broadcast_address="10.9.9.9:4567") as mocks:
            with patch(f"{_LOCK_MODULE}.StoreTicketLock") as lock_class:
                lock = create_world_ticket_lock(prefix="test/forward", poll_interval=0.5)

        lock_class.assert_called_once_with(
            store=mocks.store_class.return_value, prefix="test/forward", poll_interval=0.5
        )
        assert lock is lock_class.return_value

    def test_the_factory_result_uses_the_requested_prefix_and_poll_interval(self) -> None:
        """The real lock the factory hands back queues on the caller's keys at the caller's cadence."""
        store = _ScriptedStore(ticket=3, serve_after_polls=2, prefix="test/forward")
        clock = _FakeClock(step=0.0)

        with self._world(rank=1, broadcast_address="10.9.9.9:4567") as mocks:
            mocks.store_class.return_value = store
            lock = create_world_ticket_lock(prefix="test/forward", poll_interval=0.5)
            with patch(f"{_LOCK_MODULE}.time", clock):
                lock.acquire()

        assert store.polls == 3
        assert clock.sleeps == [0.5, 0.5]

    def test_an_abstaining_client_opens_no_store_connection(self) -> None:
        """A non-contending client still joins the broadcast but must not open a socket."""
        with self._world(rank=1, broadcast_address="10.9.9.9:4567") as mocks:
            lock = create_world_ticket_lock(prefix=PREFIX, participates=False)

        mocks.dist.broadcast_object_list.assert_called_once()
        mocks.store_class.assert_not_called()
        assert isinstance(lock, nullcontext)
