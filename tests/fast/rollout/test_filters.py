from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=[])

from miles.rollout.filter_hub.base_types import DynamicFilterOutput, FilterOutput


def test_dynamic_filter_output_is_a_compatibility_alias():
    assert DynamicFilterOutput is FilterOutput
