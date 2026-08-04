from types import SimpleNamespace

from miles.ray.wiring import launch_worker_manager


class _FakeWorkerManagerClass:
    def __init__(self) -> None:
        self.launch_calls: list[tuple[object, object]] = []
        self.handle = object()

    def launch(self, specs, pgs):
        self.launch_calls.append((specs, pgs))
        return self.handle


class TestLaunchWorkerManager:
    def test_launch_worker_manager_returns_the_manager_started_with_computed_specs_and_placements(self, monkeypatch):
        """The glue must build specs and placement groups from the same args and hand back the started manager."""
        args = SimpleNamespace(tag="run-under-test")
        specs = [object()]
        pgs = {"actor": object()}
        spec_args: list[object] = []
        pg_args: list[object] = []
        fake_manager_class = _FakeWorkerManagerClass()

        def fake_compute_specs(received_args):
            spec_args.append(received_args)
            return specs

        def fake_create_placement_groups(received_args):
            pg_args.append(received_args)
            return pgs

        monkeypatch.setattr("miles.ray.wiring.compute_specs", fake_compute_specs)
        monkeypatch.setattr("miles.ray.wiring.create_placement_groups", fake_create_placement_groups)
        monkeypatch.setattr("miles.ray.wiring.RayWorkerManager", fake_manager_class)

        result = launch_worker_manager(args)

        assert spec_args == [args]
        assert pg_args == [args]
        assert fake_manager_class.launch_calls == [(specs, pgs)]
        assert result is fake_manager_class.handle
