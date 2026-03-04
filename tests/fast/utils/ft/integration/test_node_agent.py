import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from miles.utils.ft.agents.node_agent import FtNodeAgent
from miles.utils.ft.controller.mini_prometheus import MiniPrometheus, MiniPrometheusConfig
from miles.utils.ft.models import MetricSample
from tests.fast.utils.ft.conftest import TestCollector


class TestNodeAgentMiniPrometheusIntegration:
    @pytest.mark.asyncio()
    async def test_scrape_and_instant_query(self) -> None:
        test_collector = TestCollector(
            metrics=[
                MetricSample(
                    name="gpu_temperature_celsius",
                    labels={"gpu": "0"},
                    value=72.5,
                ),
            ],
            collect_interval=0.1,
        )
        agent = FtNodeAgent(
            node_id="integ-node-0",
            collectors=[test_collector],
        )
        try:
            await agent.start()
            await asyncio.sleep(0.3)

            prom = MiniPrometheus(config=MiniPrometheusConfig())
            address = agent.get_exporter_address()
            prom.add_scrape_target(target_id="integ-node-0", address=address)
            await prom.scrape_once()

            df = prom.instant_query("miles_ft_node_gpu_temperature_celsius")
            assert not df.is_empty()
            values = df["value"].to_list()
            assert 72.5 in values
        finally:
            await agent.stop()

    @pytest.mark.asyncio()
    async def test_updated_values_visible_after_rescrape(self) -> None:
        test_collector = TestCollector(
            metrics=[
                MetricSample(
                    name="gpu_temperature_celsius",
                    labels={"gpu": "0"},
                    value=60.0,
                ),
            ],
            collect_interval=0.1,
        )
        agent = FtNodeAgent(
            node_id="integ-node-1",
            collectors=[test_collector],
        )
        try:
            await agent.start()
            await asyncio.sleep(0.3)

            prom = MiniPrometheus(config=MiniPrometheusConfig())
            address = agent.get_exporter_address()
            prom.add_scrape_target(target_id="integ-node-1", address=address)

            await prom.scrape_once()
            df1 = prom.instant_query("miles_ft_node_gpu_temperature_celsius")
            assert 60.0 in df1["value"].to_list()

            test_collector.set_metrics([
                MetricSample(
                    name="gpu_temperature_celsius",
                    labels={"gpu": "0"},
                    value=85.0,
                ),
            ])
            await asyncio.sleep(0.3)
            await prom.scrape_once()

            df2 = prom.instant_query("miles_ft_node_gpu_temperature_celsius")
            assert 85.0 in df2["value"].to_list()

            now = datetime.now(timezone.utc)
            df_range = prom.range_query(
                "miles_ft_node_gpu_temperature_celsius",
                start=now - timedelta(minutes=5),
                end=now,
                step=timedelta(seconds=10),
            )
            range_values = df_range["value"].to_list()
            assert 60.0 in range_values
            assert 85.0 in range_values
        finally:
            await agent.stop()

    @pytest.mark.asyncio()
    async def test_multiple_metrics_all_queryable(self) -> None:
        test_collector = TestCollector(
            metrics=[
                MetricSample(
                    name="gpu_temperature_celsius",
                    labels={"gpu": "0"},
                    value=70.0,
                ),
                MetricSample(
                    name="gpu_memory_used_bytes",
                    labels={"gpu": "0"},
                    value=8192.0,
                ),
                MetricSample(
                    name="gpu_power_watts",
                    labels={"gpu": "0"},
                    value=250.0,
                ),
            ],
            collect_interval=0.1,
        )
        agent = FtNodeAgent(
            node_id="integ-node-2",
            collectors=[test_collector],
        )
        try:
            await agent.start()
            await asyncio.sleep(0.3)

            prom = MiniPrometheus(config=MiniPrometheusConfig())
            address = agent.get_exporter_address()
            prom.add_scrape_target(target_id="integ-node-2", address=address)
            await prom.scrape_once()

            df_temp = prom.instant_query("miles_ft_node_gpu_temperature_celsius")
            assert 70.0 in df_temp["value"].to_list()

            df_mem = prom.instant_query("miles_ft_node_gpu_memory_used_bytes")
            assert 8192.0 in df_mem["value"].to_list()

            df_power = prom.instant_query("miles_ft_node_gpu_power_watts")
            assert 250.0 in df_power["value"].to_list()
        finally:
            await agent.stop()

    @pytest.mark.asyncio()
    async def test_label_filter_query(self) -> None:
        test_collector = TestCollector(
            metrics=[
                MetricSample(
                    name="gpu_temperature_celsius",
                    labels={"gpu": "0"},
                    value=65.0,
                ),
                MetricSample(
                    name="gpu_temperature_celsius",
                    labels={"gpu": "1"},
                    value=78.0,
                ),
            ],
            collect_interval=0.1,
        )
        agent = FtNodeAgent(
            node_id="integ-node-3",
            collectors=[test_collector],
        )
        try:
            await agent.start()
            await asyncio.sleep(0.3)

            prom = MiniPrometheus(config=MiniPrometheusConfig())
            address = agent.get_exporter_address()
            prom.add_scrape_target(target_id="integ-node-3", address=address)
            await prom.scrape_once()

            df = prom.instant_query('miles_ft_node_gpu_temperature_celsius{gpu="1"}')
            assert not df.is_empty()
            assert 78.0 in df["value"].to_list()
            assert len(df) == 1
        finally:
            await agent.stop()
