from types import SimpleNamespace

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")

import pytest

import miles.ray.multi_lora.controller as controller_mod
from miles.ray.multi_lora.backend import MultiLoRABackend
from miles.ray.multi_lora.controller import CONTROLLER_NAME, CONTROLLER_NAMESPACE, MultiLoRAController


def make_args() -> SimpleNamespace:
    return SimpleNamespace(multi_lora_n_adapters=4)


class TestConstructorContract:
    def test_keyword_arguments_build_the_backend_and_the_http_server(self):
        """Keyword construction is the supported form and wires up both owned components."""
        args = make_args()

        controller = MultiLoRAController.__ray_actor_class__(args=args, router_url="http://router:1", host="10.0.0.9")

        assert isinstance(controller.backend, MultiLoRABackend)
        assert controller.backend.router_url == "http://router:1"
        assert controller.server.host == "10.0.0.9"

    def test_positional_arguments_are_rejected(self):
        """The actor is launched by keyword only, so positional construction must fail loudly."""
        with pytest.raises(TypeError):
            MultiLoRAController.__ray_actor_class__(make_args(), "http://router:1")


class _RecordingActorClass:
    def __init__(self):
        self.options_kwargs = None
        self.positional_args = None
        self.keyword_args = None

    def options(self, **kwargs):
        self.options_kwargs = kwargs
        return self

    def remote(self, *args, **kwargs):
        self.positional_args = args
        self.keyword_args = kwargs
        return "controller-handle"


class TestCreateMultiLoRAController:
    def test_launches_a_named_head_pinned_actor_with_keyword_arguments(self, monkeypatch):
        """The factory names the actor, pins it to the head node and passes every argument by keyword."""
        recorder = _RecordingActorClass()
        monkeypatch.setattr(controller_mod, "MultiLoRAController", recorder)
        monkeypatch.setattr(controller_mod, "compute_ray_pin_head_options", lambda: {"scheduling_strategy": "head"})
        args = make_args()

        handle = controller_mod.create_multilora_controller(args, "http://router:1", "10.0.0.9")

        assert handle == "controller-handle"
        assert recorder.options_kwargs == {
            "name": CONTROLLER_NAME,
            "namespace": CONTROLLER_NAMESPACE,
            "scheduling_strategy": "head",
        }
        assert recorder.positional_args == ()
        assert recorder.keyword_args == {"args": args, "router_url": "http://router:1", "host": "10.0.0.9"}
