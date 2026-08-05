import logging
from pathlib import Path

import pytest

from miles.dashboard.sglang_scraper import DEFAULT_METRIC_WHITELIST, ScrapeMode, SglangScraper
from miles.dashboard.store import EngineSample

FIXTURES = Path(__file__).parent / "fixtures"
ROUTER_TEXT = (FIXTURES / "engine_metrics_router.txt").read_text()
DIRECT_TEXT = (FIXTURES / "engine_metrics_direct.txt").read_text()

ENGINE_A = "http://10.0.0.1:15000"
ENGINE_B = "http://10.0.0.1:15004"


def router_scraper(sink, http_get, **kwargs):
    return SglangScraper(sink, mode=ScrapeMode.ROUTER, router_addr="http://router:3000", http_get=http_get, **kwargs)


def test_router_mode_lifts_worker_addr_and_filters(caplog):
    records: list[EngineSample] = []
    urls: list[str] = []

    def fake_get(url, timeout):
        urls.append(url)
        return ROUTER_TEXT

    scraper = router_scraper(records.append, fake_get)
    count = scraper.scrape_once(now=123.0)

    assert urls == ["http://router:3000/engine_metrics"]
    assert count == len(records) == 5  # 2 running + 2 throughput + 1 token_usage; unlisted name dropped
    assert {r.metric for r in records} == {"sglang_num_running_reqs", "sglang_gen_throughput", "sglang_token_usage"}
    assert all(r.ts == 123.0 for r in records)

    running = {r.addr: r.value for r in records if r.metric == "sglang_num_running_reqs"}
    assert running == {ENGINE_A: 42.0, ENGINE_B: 7.0}

    # only the engine_type label survives; model_name and worker_addr are dropped
    token_usage = next(r for r in records if r.metric == "sglang_token_usage")
    assert token_usage.labels == {"engine_type": "prefill"}
    assert all("worker_addr" not in r.labels and "model_name" not in r.labels for r in records)


def test_direct_mode_normalizes_colon_names_and_attaches_addr():
    records: list[EngineSample] = []
    scraper = SglangScraper(
        records.append,
        mode=ScrapeMode.DIRECT,
        engine_addrs=lambda: [ENGINE_A],
        http_get=lambda url, timeout: DIRECT_TEXT if url == f"{ENGINE_A}/metrics" else pytest.fail(url),
    )
    count = scraper.scrape_once(now=5.0)

    assert count == 2  # uptime is filtered out
    assert {(r.addr, r.metric, r.value) for r in records} == {
        (ENGINE_A, "sglang_num_running_reqs", 13.0),
        (ENGINE_A, "sglang_cache_hit_rate", 0.71),
    }


def test_direct_mode_skips_failed_engine(caplog):
    records: list[EngineSample] = []

    def flaky_get(url, timeout):
        if ENGINE_B in url:
            raise ConnectionError("engine restarting")
        return DIRECT_TEXT

    scraper = SglangScraper(
        records.append,
        mode=ScrapeMode.DIRECT,
        engine_addrs=lambda: [ENGINE_A, ENGINE_B],
        http_get=flaky_get,
    )
    with caplog.at_level(logging.WARNING):
        count = scraper.scrape_once(now=1.0)

    assert count == 2  # engine A's samples still land
    assert {r.addr for r in records} == {ENGINE_A}
    assert len([r for r in caplog.records if "skipping this tick" in r.message]) == 1


def test_warnings_are_rate_limited(caplog):
    scraper = SglangScraper(
        lambda r: None,
        mode=ScrapeMode.DIRECT,
        engine_addrs=lambda: [ENGINE_A],
        http_get=lambda url, timeout: (_ for _ in ()).throw(ConnectionError("down")),
    )
    with caplog.at_level(logging.WARNING):
        scraper.scrape_once(now=1.0)
        scraper.scrape_once(now=2.0)  # within the suppression window
    assert len(caplog.records) == 1

    scraper._warner.reset_window_for_test()  # window elapses
    with caplog.at_level(logging.WARNING):
        scraper.scrape_once(now=3.0)
    assert len(caplog.records) == 2


def test_router_sample_without_worker_addr_is_dropped(caplog):
    records: list[EngineSample] = []
    text = '# TYPE sglang_num_running_reqs gauge\nsglang_num_running_reqs{model_name="q"} 3.0\n'
    scraper = router_scraper(records.append, lambda url, timeout: text)
    with caplog.at_level(logging.WARNING):
        assert scraper.scrape_once(now=1.0) == 0
    assert records == []
    assert "lacks worker_addr" in caplog.records[0].message


def test_whitelist_override():
    records: list[EngineSample] = []
    scraper = router_scraper(records.append, lambda url, timeout: ROUTER_TEXT, whitelist=("sglang_gen_throughput",))
    scraper.scrape_once(now=1.0)
    assert {r.metric for r in records} == {"sglang_gen_throughput"}


def test_mode_prerequisites():
    with pytest.raises(AssertionError, match="router_addr"):
        SglangScraper(lambda r: None, mode=ScrapeMode.ROUTER)
    with pytest.raises(AssertionError, match="engine_addrs"):
        SglangScraper(lambda r: None, mode=ScrapeMode.DIRECT)


def test_default_whitelist_covers_design_set():
    assert "sglang_num_running_reqs" in DEFAULT_METRIC_WHITELIST
    assert "sglang_kv_transfer_speed_gb_s" in DEFAULT_METRIC_WHITELIST  # PD set present
