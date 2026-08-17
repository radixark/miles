"""InferenceAdminPort contract: the backend invokes init()/close() as part of
its lifecycle, so the port must declare them — a fake implementing exactly the
declared protocol must never surprise the backend with an AttributeError
(external review)."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")

from miles.ray.tinker_backend.inference_admin import InferenceAdminPort, RouterInferenceAdmin


def test_declared_port_includes_the_invoked_lifecycle():
    for method in ("init", "close", "abort_registration"):
        assert hasattr(InferenceAdminPort, method), f"InferenceAdminPort must declare {method}()"


def test_the_router_concrete_satisfies_the_declared_surface():
    admin = RouterInferenceAdmin("http://router:1")
    for method in ("init", "close", "abort_registration"):
        assert callable(getattr(admin, method))
