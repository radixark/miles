from miles.utils.ft.controller.metrics.mini_prometheus.scraper import parse_prometheus_text


class TestParsePrometheusText:
    """Smoke tests for the parse_prometheus_text wrapper.

    Detailed parsing behaviour is covered by prometheus_client itself;
    these tests verify our wrapper returns MetricSample objects correctly.
    """

    def test_simple_metric(self) -> None:
        text = "# TYPE gpu_temp gauge\ngpu_temp 75.0\n"
        samples = parse_prometheus_text(text)
        assert len(samples) == 1
        assert samples[0].name == "gpu_temp"
        assert samples[0].value == 75.0
        assert samples[0].labels == {}

    def test_metric_with_labels(self) -> None:
        text = '# TYPE gpu_temp gauge\ngpu_temp{gpu="0",node="n1"} 82.5\n'
        samples = parse_prometheus_text(text)
        assert len(samples) == 1
        assert samples[0].labels == {"gpu": "0", "node": "n1"}
        assert samples[0].value == 82.5

    def test_multiple_metrics(self) -> None:
        text = "# TYPE metric_a gauge\n" "metric_a 1.0\n" "# TYPE metric_b gauge\n" "metric_b 2.0\n"
        samples = parse_prometheus_text(text)
        assert len(samples) == 2

    def test_empty_input_returns_empty(self) -> None:
        assert parse_prometheus_text("") == []
