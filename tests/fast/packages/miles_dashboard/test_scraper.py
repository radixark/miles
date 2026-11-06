import logging
from pathlib import Path

import pytest
from prometheus_client.parser import text_string_to_metric_families

from miles.dashboard.sglang_scraper import DEFAULT_METRIC_WHITELIST, SglangScraper
from miles.dashboard.store import EngineSample

FIXTURES = Path(__file__).parent / "fixtures"
DIRECT_TEXT = (FIXTURES / "engine_metrics_direct.txt").read_text()

ENGINE_A = "http://10.0.0.1:15000"
ENGINE_B = "http://10.0.0.1:15004"


def test_colon_names_are_normalized_and_the_engine_addr_attached():
    records: list[EngineSample] = []
    scraper = SglangScraper(
        records.append,
        engine_addrs=lambda: [ENGINE_A],
        http_get=lambda url, timeout: DIRECT_TEXT if url == f"{ENGINE_A}/metrics" else pytest.fail(url),
    )
    count = scraper.scrape_once(now=5.0)

    assert count == 2  # uptime is filtered out
    assert {(r.addr, r.metric, r.value) for r in records} == {
        (ENGINE_A, "sglang_num_running_reqs", 13.0),
        (ENGINE_A, "sglang_cache_hit_rate", 0.71),
    }


def test_a_failed_engine_is_skipped(caplog):
    records: list[EngineSample] = []

    def flaky_get(url, timeout):
        if ENGINE_B in url:
            raise ConnectionError("engine restarting")
        return DIRECT_TEXT

    scraper = SglangScraper(
        records.append,
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


def test_whitelist_override():
    records: list[EngineSample] = []
    scraper = SglangScraper(
        records.append,
        engine_addrs=lambda: [ENGINE_A],
        http_get=lambda url, timeout: DIRECT_TEXT,
        whitelist=("sglang_cache_hit_rate",),
    )
    scraper.scrape_once(now=1.0)
    assert {r.metric for r in records} == {"sglang_cache_hit_rate"}


def test_default_whitelist_covers_design_set():
    assert "sglang_num_running_reqs" in DEFAULT_METRIC_WHITELIST
    assert "sglang_kv_transfer_speed_gb_s" in DEFAULT_METRIC_WHITELIST  # PD set present


def test_dp_rank_label_survives():
    records: list[EngineSample] = []
    text = (
        "# TYPE sglang:num_running_reqs gauge\n"
        'sglang:num_running_reqs{model_name="q",engine_type="unified",dp_rank="0",tp_rank="0"} 11.0\n'
        'sglang:num_running_reqs{model_name="q",engine_type="unified",dp_rank="1",tp_rank="1"} 1.0\n'
    )
    scraper = SglangScraper(records.append, engine_addrs=lambda: [ENGINE_A], http_get=lambda url, timeout: text)
    assert scraper.scrape_once(now=1.0) == 2
    assert {(r.labels["dp_rank"], r.value) for r in records} == {("0", 11.0), ("1", 1.0)}
    assert all("tp_rank" not in r.labels and "model_name" not in r.labels for r in records)


@pytest.mark.parametrize(
    "page",
    ["engine_metrics_dp_attention.txt", "engine_metrics_direct.txt"],
    ids=lambda p: p.split(".")[0],
)
def test_prefiltered_page_yields_what_the_whole_page_does(page):
    """Dropping non-whitelisted lines before parsing must not change a single
    sample, histogram _sum/_count and multi-rank labels included."""

    text = (FIXTURES / page).read_text()
    whitelist = frozenset(DEFAULT_METRIC_WHITELIST)
    expected = [
        (
            sample.name.replace(":", "_"),
            {k: v for k, v in sample.labels.items() if k in ("engine_type", "dp_rank")},
            sample.value,
        )
        for family in text_string_to_metric_families(text)
        for sample in family.samples
        if sample.name.replace(":", "_") in whitelist
    ]
    got = []
    scraper = SglangScraper(got.append, engine_addrs=lambda: [])
    scraper._ingest("http://e:1", text, now=1.0)
    assert [(s.metric, s.labels, s.value) for s in got] == expected
    assert expected, "fixture has no whitelisted samples"


def test_prefilter_keeps_a_label_holding_a_unicode_line_break():
    """str.splitlines also breaks on \x85 and
    ; the parser splits on "\\n"
        only, so such a label value must reach it whole."""

    text = 'sglang:num_running_reqs{model_name="a\x85b c"} 3.0\nsglang:other_metric 1.0\n'
    scraper = SglangScraper(lambda sample: None, engine_addrs=lambda: [])
    [family] = text_string_to_metric_families(scraper._whitelisted_lines(text))
    [sample] = family.samples
    assert sample.labels["model_name"] == "a\x85b c" and sample.value == 3.0
