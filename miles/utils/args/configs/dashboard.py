from miles.utils.args.schema import A, Arg, BaseConfig


class DashboardConfig(BaseConfig):
    use_miles_dashboard: A[
        bool,
        Arg(
            help="Collect dashboard telemetry (phases, GPU util, engine metrics) under {dump-details}/dashboard/. "
            "Requires --dump-details. View with `python -m miles.dashboard.serve`.",
        ),
    ] = False
    dashboard_flush_interval: A[float, Arg(help="collector disk flush cadence (s)")] = 5.0
    dashboard_gpu_sample_interval: A[float, Arg(help="NVML sampling cadence (s)")] = 1.0
    dashboard_sglang_scrape_interval: A[float, Arg(help="engine scrape cadence (s)")] = 2.0
    dashboard_sglang_scrape_mode: A[
        str,
        Arg(
            choices=["auto", "router", "direct"],
            help="auto scrapes each engine's /metrics; router scrapes {router}/engine_metrics",
        ),
    ] = "auto"
    dashboard_sglang_metrics: A[
        str | None,
        Arg(help="comma-separated override of the scraped sglang metric whitelist"),
    ] = None
    dashboard_forward_prometheus: A[
        bool,
        Arg(help="also push dashboard gauges to the --use-prometheus collector for external Grafana"),
    ] = False
