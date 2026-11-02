import errno
import socket
import struct
import sys
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from miles.utils import port_range
from miles.utils.misc import get_free_port
from miles.utils.port_range import ephemeral_port_range, ray_worker_port_args, service_port_range

# Linux 6.3+: narrows one socket's ephemeral range, packed as low | high << 16.
_IP_LOCAL_PORT_RANGE = 51


@pytest.fixture
def ephemeral_range_file(tmp_path, monkeypatch):
    """Point the module at a stand-in for /proc/sys/net/ipv4/ip_local_port_range."""
    path = tmp_path / "ip_local_port_range"
    monkeypatch.setattr(port_range, "_EPHEMERAL_PORT_RANGE_FILE", path)

    def write(low: int, high: int) -> None:
        path.write_text(f"{low}\t{high}\n")

    return write


class TestEphemeralPortRange:
    def test_reads_the_kernel_format(self, ephemeral_range_file):
        """The kernel separates the two bounds with a tab and ends with a newline."""
        ephemeral_range_file(9000, 65000)

        assert ephemeral_port_range() == (9000, 65000)

    def test_an_unreadable_range_is_unknown(self, ephemeral_range_file):
        """Hosts without procfs, such as macOS, have no range to read."""
        assert ephemeral_port_range() is None


class TestServicePortRange:
    def test_the_default_kernel_range_keeps_the_usual_start(self, ephemeral_range_file):
        """With Linux's default 32768-60999 nothing moves, but the range now ends where the ephemeral one begins."""
        ephemeral_range_file(32768, 60999)

        assert service_port_range() == (20000, 32767)

    def test_a_range_that_covers_20000_moves_below_it(self, ephemeral_range_file):
        """Hosts whose ephemeral range starts at 9000 keep service ports under 9000."""
        ephemeral_range_file(9000, 65000)

        assert service_port_range() == (1024, 8999)

    def test_the_stretch_below_stays_clear_of_rays_default_worker_ports(self, ephemeral_range_file):
        """Ray hands out 10002-19999 by default, so the low stretch ends before them."""
        ephemeral_range_file(15000, 60999)

        assert service_port_range() == (1024, 10001)

    def test_the_wider_stretch_wins(self, ephemeral_range_file):
        """With almost nothing free below the ephemeral range, the stretch above it is used."""
        ephemeral_range_file(1100, 40000)

        assert service_port_range() == (40001, 65535)

    def test_a_range_below_20000_leaves_the_usual_range_whole(self, ephemeral_range_file):
        """The kernel hands out nothing from 20000 up, so the range runs to the last port."""
        ephemeral_range_file(1024, 15000)

        assert service_port_range() == (20000, 65535)

    def test_a_range_with_no_room_outside_it_falls_back_to_the_default(self, ephemeral_range_file, caplog):
        """There is nothing safe to choose, so allocation behaves as it did before and says why."""
        ephemeral_range_file(1024, 65535)

        with caplog.at_level("WARNING", logger=port_range.__name__):
            assert service_port_range() == (20000, 65535)

        assert "ephemeral port range 1024-65535" in caplog.text

    def test_an_unknown_range_keeps_the_default(self, ephemeral_range_file):
        assert service_port_range() == (20000, 65535)


class TestRayWorkerPortArgs:
    def test_the_default_kernel_range_keeps_rays_worker_ports(self, ephemeral_range_file):
        """Ray's 10002-19999 already lies below 32768, so `ray start` is left unchanged."""
        ephemeral_range_file(32768, 60999)

        assert ray_worker_port_args() == ""

    def test_an_overlapping_range_lets_the_kernel_assign_worker_ports(self, ephemeral_range_file):
        """Port 0 makes each worker bind a kernel-assigned port, so there is no gap to lose it in."""
        ephemeral_range_file(9000, 65000)

        assert ray_worker_port_args() == " --min-worker-port 0 --max-worker-port 0"

    def test_an_unknown_range_keeps_rays_worker_ports(self, ephemeral_range_file):
        assert ray_worker_port_args() == ""


@contextmanager
def _outgoing_connection_asking_for(port: int) -> Iterator[int]:
    """Open a loopback connection that asks the kernel for `port` as its source port, and yield the port it got.

    The kernel honours the request only inside the host's ephemeral range and otherwise picks
    from the whole range, so the connection gets `port` exactly when any outgoing connection
    could be handed it.
    """
    with socket.socket() as listener, socket.socket() as client:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        try:
            client.setsockopt(socket.IPPROTO_IP, _IP_LOCAL_PORT_RANGE, struct.pack("I", port | port << 16))
        except OSError:
            pytest.skip("the kernel has no IP_LOCAL_PORT_RANGE")
        client.connect(listener.getsockname())
        yield client.getsockname()[1]


def _serve_on(port: int) -> None:
    """Bind and listen on `port` the way a service handed that port does."""
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("", port))
        server.listen()


@pytest.mark.skipif(sys.platform != "linux", reason="needs the Linux ephemeral port range")
class TestProbedPortsAndOutgoingConnections:
    """A probed port is bound only later, and an outgoing connection opened in between can take it.

    In production the kernel picks such a source port at random from the ephemeral range, so
    the collision is rare per connection but certain over enough of them. Here one connection
    asks for the probed port, which makes the outcome deterministic.
    """

    def test_a_probed_port_inside_the_ephemeral_range_can_be_taken_before_it_is_bound(self):
        """This is the startup failure: the port was free when probed and in use when served."""
        ephemeral = ephemeral_port_range()
        if ephemeral is None:
            pytest.skip("the ephemeral port range is unreadable")
        port = get_free_port(start_port=ephemeral[0], port_range=ephemeral)

        with _outgoing_connection_asking_for(port) as source_port:
            assert source_port == port
            with pytest.raises(OSError) as error:
                _serve_on(port)

        assert error.value.errno == errno.EADDRINUSE

    def test_a_probed_service_port_cannot_be_taken(self):
        """The kernel never hands out a port from the service range as a source port."""
        ephemeral = ephemeral_port_range()
        low, high = service_port_range()
        if ephemeral is None or (low <= ephemeral[1] and ephemeral[0] <= high):
            pytest.skip("this host has no service range outside its ephemeral range")
        port = get_free_port(start_port=low, port_range=(low, high))

        with _outgoing_connection_asking_for(port) as source_port:
            assert source_port != port
            _serve_on(port)
