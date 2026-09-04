import pytest

pytest.importorskip("sglang")

from miles.backends.sglang_utils.sglang_engine import build_server_url, format_v6_uri  # noqa: E402


class TestFormatV6Uri:
    @pytest.mark.parametrize("addr", ["10.0.0.1", "localhost", "my-host.internal"])
    def test_it_leaves_non_v6_addresses_alone(self, addr):
        """Bracketing a v4 address or a hostname would break the url."""
        assert format_v6_uri(addr) == addr

    def test_it_brackets_a_bare_v6_address(self):
        """A bare v6 address makes ``host:port`` ambiguous."""
        assert format_v6_uri("::1") == "[::1]"

    def test_it_is_idempotent(self):
        """The engine already brackets its host; the rollout process starts from the bare
        one out of the addr allocator, so both must land on the same url."""
        assert format_v6_uri("[::1]") == "[::1]"
        assert format_v6_uri(format_v6_uri("::1")) == "[::1]"

    @pytest.mark.parametrize("addr", [None, ""])
    def test_it_passes_a_missing_address_through(self, addr):
        """The host is optional until the engine resolves its own."""
        assert format_v6_uri(addr) == addr


class TestBuildServerUrl:
    def test_it_builds_the_same_url_from_a_bracketed_or_bare_v6_host(self):
        """Two processes derive this url from different sources; they must agree."""
        assert build_server_url("::1", 30000) == "http://[::1]:30000"
        assert build_server_url("[::1]", 30000) == "http://[::1]:30000"

    def test_it_builds_a_plain_v4_url(self):
        """The common case must stay byte-identical to the old f-string."""
        assert build_server_url("10.0.0.1", 30000) == "http://10.0.0.1:30000"
