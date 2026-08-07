import pytest
import yaml
from pydantic import ValidationError
from tests.fast.utils.workers.worker_provider.kubernetes.run_specs import RELEASE, make_engine_spec, make_trainer_spec

from miles.utils.external_utils.command_utils.helm_backend import elastic
from miles.utils.external_utils.command_utils.helm_backend import values as values_module
from miles.utils.workers.worker_spec import (
    BaseWorkerSpec,
    CommandWorkerSpec,
    PortInfo,
    SchedulingSpec,
    ServeWorkerSpec,
)

LAYOUT = values_module.RunLayout(
    run_id="260101-000000-000",
    release="r",
    orchestrator_command=["python", "train.py"],
    worker_argv=["--foo", "bar"],
)


def _router() -> CommandWorkerSpec:
    return CommandWorkerSpec(
        name="inference-router-0",
        port_infos=[PortInfo(name="primary", static_port=8000)],
        env_var=lambda ctx: {},
        scheduling=SchedulingSpec.single(num_gpus_per_worker=0),
        launch_command=lambda ctx: f"python -m router --host {ctx.self_addrs['primary'].host}",
    )


def _engine(
    num_cells: int = 2,
    gpus_per_engine: int = 32,
    name: str = "inference-engine-0-0",
    colocate_with_trainer: bool = False,
) -> CommandWorkerSpec:
    return CommandWorkerSpec(
        name=name,
        port_infos=[
            PortInfo(name="primary", static_port=8000),
            PortInfo(name="dist_init", static_port=9000, mode="master"),
            PortInfo(name="engine_info_bootstrap", static_port=12000),
        ],
        env_var=lambda ctx: {"NVSHMEM_DISABLE_NCCL": "1"},
        scheduling=SchedulingSpec(
            num_cells=num_cells,
            num_workers_per_cell=max(1, gpus_per_engine // 8),
            num_gpus_per_worker=0.2,
            num_gpu_slots_per_worker=min(gpus_per_engine, 8),
            num_gpus_per_node=8,
            colocate_with_trainer=colocate_with_trainer,
        ),
        launch_command=lambda ctx: (
            f"python -m sglang.launch_server --node-rank {ctx.worker_in_cell_index} "
            f"--dist-init-addr {ctx.self_addrs['dist_init'].host}:{ctx.self_addrs['dist_init'].port} "
            f"--base-gpu-id {ctx.gpu_ids[0]}"
        ),
    )


def _trainer(num_cells: int = 2, gpus_per_cell: int = 16) -> ServeWorkerSpec:
    return ServeWorkerSpec(
        name="trainer-actor",
        port_infos=[PortInfo(name="master", static_port=9000, mode="master")],
        env_var=lambda ctx: {"NCCL_CUMEM_ENABLE": "0"},
        scheduling=SchedulingSpec(
            num_cells=num_cells,
            num_workers_per_cell=gpus_per_cell,
            num_gpus_per_worker=0.4,
            num_gpu_slots_per_worker=1,
            num_gpus_per_node=8,
        ),
        worker_class="miles.backends.megatron_utils.actor.MegatronTrainRayActor",
        ctor_kwargs=lambda ctx: {},
    )


def _session_server(num_cells: int) -> CommandWorkerSpec:
    return CommandWorkerSpec(
        name="session-server",
        port_infos=[PortInfo(name="primary", static_port=8000)],
        env_var=lambda ctx: {},
        scheduling=SchedulingSpec(
            num_cells=num_cells, num_workers_per_cell=1, num_gpus_per_worker=0, num_gpu_slots_per_worker=0
        ),
        launch_command=lambda ctx: "python -m session_server",
    )


def _session_client() -> CommandWorkerSpec:
    return CommandWorkerSpec(
        name="rollout-executor",
        port_infos=[PortInfo(name="primary", static_port=8100)],
        env_var=lambda ctx: {},
        scheduling=SchedulingSpec.single(num_gpus_per_worker=0),
        launch_command=lambda ctx: "python -m executor --session-servers "
        + ",".join(addrs["primary"].addr for addrs in ctx.spec_addrs["session-server"]),
    )


class TestSectionOf:
    def test_sends_a_gpu_less_spec_to_the_static_workers(self):
        """A router is never healed per cell, so a pool_id would only add indirection."""
        assert values_module.section_of(_router()) == "staticWorkers"

    def test_keeps_a_single_cell_engine_a_pool(self):
        """The provider recognises cells by LeaderWorkerSet labels, which a plain workload would not carry."""
        assert values_module.section_of(_engine(num_cells=1, gpus_per_engine=8)) == "inferenceEngines"

    def test_sends_a_command_pool_to_the_engines(self):
        """An engine group is restarted as a unit and so needs a pool_id."""
        assert values_module.section_of(_engine()) == "inferenceEngines"

    def test_sends_a_serve_pool_to_the_trainers(self):
        """Trainers are served over rpc rather than launched as a command, and heal per dp group."""
        assert values_module.section_of(_trainer()) == "trainers"


class TestRunLayout:
    def test_rejects_a_field_the_chart_would_never_read(self):
        """A misspelled layout field would otherwise be dropped, and the run would launch mis-shaped."""
        with pytest.raises(ValidationError):
            values_module.RunLayout(
                run_id="260101-000000-000",
                release="r",
                orchestrator_command=[],
                worker_argv=[],
                node_local_rooot="/scratch",
            )


class TestBuildValues:
    def test_keeps_everything_it_computes_under_one_key(self):
        """The infra values own every other top-level key, and a run must not be able to reach them."""
        assert list(values_module.build_values([], LAYOUT)) == ["run"]

    def test_records_the_run_id_the_run_directory_is_named_after(self):
        """The wrapper writes its exit file under this id, and the launcher polls the same path."""
        assert values_module.build_values([], LAYOUT)["run"]["id"] == "260101-000000-000"

    def test_hands_the_orchestrator_the_command_that_starts_the_run(self):
        """Nothing else launches training, so an empty orchestrator command is a run that never starts."""
        built = values_module.build_values([], LAYOUT)

        assert built["run"]["orchestrator"] == {"command": ["python", "train.py"]}

    def test_files_every_spec_under_its_own_section(self):
        """Each section renders a different workload kind, so a misfiled spec deploys the wrong shape."""
        built = values_module.build_values([_router(), _engine(), _trainer()], LAYOUT)["run"]

        assert [entry["name"] for entry in built["staticWorkers"]] == ["inference-router-0"]
        assert [entry["name"] for entry in built["inferenceEngines"]] == ["inference-engine-0-0"]
        assert [entry["name"] for entry in built["trainers"]] == ["trainer-actor"]

    def test_deploys_nothing_for_a_spec_a_run_has_turned_off(self):
        """A session server the run disabled asks for no cells, and a pod for it would serve nobody."""
        built = values_module.build_values([_router(), _session_server(num_cells=0)], LAYOUT)["run"]

        assert [entry["name"] for entry in built["staticWorkers"]] == ["inference-router-0"]

    def test_leaves_the_addresses_of_a_disabled_spec_out_of_its_readers_commands(self):
        """A url of a pod nobody deploys would send every request of the run into a black hole."""
        built = values_module.build_values([_session_server(num_cells=0)], LAYOUT)["run"]

        assert built["staticWorkers"] == []

    def test_refuses_to_build_an_entry_for_a_spec_that_wants_no_cell(self):
        """Every entry renders at least one pod, so a zero-cell spec reaching conversion is a silent deploy."""
        with pytest.raises(AssertionError, match="cells"):
            values_module._build_entry(_session_server(num_cells=0), layout=LAYOUT, addresses={})

    def test_gives_a_single_cell_pool_the_replica_count_it_asks_for(self):
        """An omitted count reads as an unmanaged field, and elastic then refuses to scale the pool_id at all."""
        built = values_module.build_values([_engine(num_cells=1, gpus_per_engine=8), _trainer(num_cells=1)], LAYOUT)

        assert built["run"]["inferenceEngines"][0]["replicas"] == 1
        assert built["run"]["trainers"][0]["replicas"] == 1

    def test_gives_a_static_worker_one_instance_per_cell(self):
        """Two session servers are two addresses, and one statefulset pod could answer only one of them."""
        built = values_module.build_values([_session_server(num_cells=2)], LAYOUT)["run"]

        assert built["staticWorkers"][0]["replicas"] == 2

    def test_addresses_each_static_cell_by_its_own_hostname(self):
        """The rollout executor resolves a list of session server urls, which pod zero cannot all be."""
        built = values_module.build_values([_session_server(num_cells=2), _session_client()], LAYOUT)["run"]
        command = built["staticWorkers"][1]["command"]

        assert command[-1] == (
            "http://r-miles-run-session-server-0.r-miles-run-session-server:8000,"
            "http://r-miles-run-session-server-1.r-miles-run-session-server:8000"
        )

    def test_sizes_an_engine_group_by_its_nodes(self):
        """A 32-gpu engine spans four 8-gpu nodes, which is four pods in one group."""
        engine = values_module.build_values([_engine()], LAYOUT)["run"]["inferenceEngines"][0]

        assert (engine["replicas"], engine["size"]) == (2, 4)
        assert engine["resources"] == {"limits": {"nvidia.com/gpu": 8}}

    def test_replaces_only_the_node_rank_with_the_kubelet_placeholder(self):
        """Every pod of a group shares one command, so the rank must be the one part left to kubelet."""
        command = values_module.build_values([_engine()], LAYOUT)["run"]["inferenceEngines"][0]["command"]

        assert command[command.index("--node-rank") + 1] == values_module.WORKER_INDEX_PLACEHOLDER
        assert command[command.index("--base-gpu-id") + 1] == "0"

    def test_points_a_master_port_at_the_group_leader(self):
        """A rank cannot know its leader's address until it is scheduled, but kubelet does."""
        command = values_module.build_values([_engine()], LAYOUT)["run"]["inferenceEngines"][0]["command"]

        assert command[command.index("--dist-init-addr") + 1] == f"{values_module.LEADER_ADDRESS_PLACEHOLDER}:9000"

    def test_binds_a_pod_own_listener_on_every_interface(self):
        """A pod has its own network namespace, so it need not guess which address it will get."""
        worker = values_module.build_values([_router()], LAYOUT)["run"]["staticWorkers"][0]

        assert worker["command"][worker["command"].index("--host") + 1] == "0.0.0.0"

    def test_supervises_one_trainer_rank_per_gpu_in_the_pod(self):
        """A trainer worker is a rank, and a pod holding eight gpus has to run eight of them."""
        trainer = values_module.build_values([_trainer()], LAYOUT)["run"]["trainers"][0]

        assert trainer["command"][trainer["command"].index("--num-subprocesses") + 1] == "8"
        assert (trainer["replicas"], trainer["size"]) == (2, 2)

    def test_passes_the_run_argv_to_every_served_worker(self):
        """A pod rebuilds the run's arguments itself rather than receiving a pickled namespace."""
        trainer = values_module.build_values([_trainer()], LAYOUT)["run"]["trainers"][0]

        assert trainer["command"][-2:] == ["--foo", "bar"]

    def test_serves_a_single_rank_trainer_without_a_supervisor(self):
        """One rank needs no supervisor, and adding one would bury its logs a level deeper."""
        built = values_module.build_values([_trainer(num_cells=1, gpus_per_cell=1)], LAYOUT)["run"]

        assert "--num-subprocesses" not in built["trainers"][0]["command"]

    def test_renames_ports_kubernetes_would_reject(self):
        """A port name may not hold an underscore and stops at fifteen characters."""
        engine = values_module.build_values([_engine()], LAYOUT)["run"]["inferenceEngines"][0]

        assert [port["name"] for port in engine["ports"]] == ["primary", "dist-init", "engine-info-boo"]

    def test_a_served_pool_carries_no_environment_of_its_own(self):
        """A trainer rank's env depends on which rank it is, which no values entry can know."""
        trainer = values_module.build_values([_trainer()], LAYOUT)["run"]["trainers"][0]

        assert "env" not in trainer
        assert "--specs" in trainer["command"]

    def test_refuses_a_command_pool_whose_environment_depends_on_its_rank(self):
        """Such a value would be rendered once for the whole pool and then be wrong for every cell but one."""
        spec = _engine().model_copy(update={"env_var": lambda ctx: {"HOME_OF": str(ctx.cell_index)}})

        with pytest.raises(AssertionError, match="builds its environment out of the cell and rank"):
            values_module.build_values([spec], LAYOUT)

    def test_keeps_the_environment_a_command_pool_states_outright(self):
        """An engine's env is the same for every cell, so the chart may carry it."""
        engine = values_module.build_values([_engine()], LAYOUT)["run"]["inferenceEngines"][0]

        assert engine["env"] == {"NVSHMEM_DISABLE_NCCL": "1"}

    def test_omits_an_empty_environment(self):
        """An empty env block in the values would render an empty env list into the pod."""
        assert "env" not in values_module.build_values([_router()], LAYOUT)["run"]["staticWorkers"][0]

    def test_rejects_a_cell_that_is_not_a_whole_number_of_nodes(self):
        """Such a cell would leave its trailing ranks with no pod to run in."""
        spec = _engine().model_copy(
            update={
                "scheduling": _engine().scheduling.model_copy(
                    update={"num_workers_per_cell": 3, "num_gpu_slots_per_worker": 5}
                )
            }
        )

        with pytest.raises(AssertionError, match="whole number"):
            values_module.build_values([spec], LAYOUT)

    def test_allows_a_command_that_never_mentions_its_rank(self):
        """Some engines do not take one; what matters is that no sentinel survives into the command."""
        spec = _engine().model_copy(update={"launch_command": lambda ctx: "python -m sglang.launch_server"})

        command = values_module.build_values([spec], LAYOUT)["run"]["inferenceEngines"][0]["command"]

        assert str(values_module._WORKER_INDEX_SENTINEL) not in " ".join(command)


COLOCATE_LAYOUT = LAYOUT.model_copy(update={"colocate": True})


def _disaggregated_engines(*, colocated: str | None) -> list[CommandWorkerSpec]:
    return [
        _engine(
            num_cells=4,
            gpus_per_engine=8,
            name=name,
            colocate_with_trainer=name == colocated,
        )
        for name in ("inference-engine-0-0", "inference-engine-0-1")
    ]


class TestColocateSection:
    def test_pairs_the_pool_a_disaggregated_run_declares(self):
        """Prefill and decode are the same shape, so only the run itself knows which one shares the gpus."""
        specs = [*_disaggregated_engines(colocated="inference-engine-0-1"), _trainer(num_cells=4, gpus_per_cell=8)]

        built = values_module.build_values(specs, COLOCATE_LAYOUT)["run"]

        assert built["colocate"] == {
            "enabled": True,
            "enginePool": "inference-engine-0-1",
            "trainerPool": "trainer-actor",
        }

    def test_refuses_a_disaggregated_run_that_names_no_pool(self):
        """Guessing the pool_id from its shape found none of a prefill/decode pair, and silently so."""
        specs = [*_disaggregated_engines(colocated=None), _trainer(num_cells=4, gpus_per_cell=8)]

        with pytest.raises(AssertionError, match="--colocate-engine-pool"):
            values_module.build_values(specs, COLOCATE_LAYOUT)

    def test_refuses_two_pools_that_both_claim_the_trainer(self):
        """One pairing controller pins one pool_id, so a second claim would leave half the engines adrift."""
        specs = [
            _engine(num_cells=4, gpus_per_engine=8, name="inference-engine-0-0", colocate_with_trainer=True),
            _engine(num_cells=4, gpus_per_engine=8, name="inference-engine-0-1", colocate_with_trainer=True),
            _trainer(num_cells=4, gpus_per_cell=8),
        ]

        with pytest.raises(AssertionError, match="--colocate-engine-pool"):
            values_module.build_values(specs, COLOCATE_LAYOUT)

    def test_pairs_a_decode_pool_that_leaves_trainer_gpus_to_themselves(self):
        """Prefill runs on its own nodes, so the colocated decode pool_id covers only part of the trainer."""
        specs = [
            _engine(num_cells=2, gpus_per_engine=8, colocate_with_trainer=True),
            _trainer(num_cells=1, gpus_per_cell=32),
        ]

        built = values_module.build_values(specs, COLOCATE_LAYOUT)["run"]

        assert built["colocate"]["enginePool"] == "inference-engine-0-0"

    def test_refuses_more_engine_cells_than_the_trainer_can_seat(self):
        """An engine rank on a gpu no trainer shares would receive nothing from a weight update."""
        specs = [
            _engine(num_cells=8, gpus_per_engine=8, colocate_with_trainer=True),
            _trainer(num_cells=1, gpus_per_cell=32),
        ]

        with pytest.raises(AssertionError, match="do not fit"):
            values_module.build_values(specs, COLOCATE_LAYOUT)

    def test_rejects_a_declared_pool_whose_cell_is_smaller_than_a_node(self):
        """The device plugin picks the cards, so a sub-node engine's base gpu id cannot be rendered."""
        specs = [
            _engine(num_cells=1, gpus_per_engine=4, colocate_with_trainer=True),
            _trainer(num_cells=1, gpus_per_cell=4),
        ]

        with pytest.raises(AssertionError, match="sub-node"):
            values_module.build_values(specs, COLOCATE_LAYOUT)

    def test_leaves_a_run_that_does_not_colocate_without_the_section(self):
        """A disaggregated run must not gain a pairing controller with pod write rights."""
        specs = [*_disaggregated_engines(colocated=None), _trainer(num_cells=4, gpus_per_cell=8)]

        assert "colocate" not in values_module.build_values(specs, LAYOUT)["run"]


STAGING_LAYOUT = values_module.RunLayout(
    run_id="260101-000000-000",
    release="r",
    orchestrator_command=["python", "train.py"],
    worker_argv=["--foo", "bar"],
    stage_to_local=("/cluster-storage/dataset:/scratch/dataset",),
    node_local_root="/scratch",
)


def _staged(section: str, spec: BaseWorkerSpec) -> list[str]:
    return values_module.build_values([spec], STAGING_LAYOUT)["run"][section][0]["command"]


class TestStageToLocal:
    def test_a_trainer_copies_its_inputs_to_the_node_before_it_starts(self):
        """Every trainer rank reads the dataset every step, and the shared filesystem cannot serve that."""
        command = _staged("trainers", _trainer())

        assert command[:2] == ["bash", "-c"]
        assert "/cluster-storage/dataset" in command[2]
        assert "/scratch/dataset" in command[2]

    def test_an_engine_is_launched_without_a_staging_step(self):
        """An sglang engine loads weights the trainer sends it, so staging would copy gigabytes nothing reads."""
        command = _staged("inferenceEngines", _engine())

        assert command[:2] != ["bash", "-c"]
        assert "/scratch/dataset" not in " ".join(command)

    def test_a_static_worker_is_launched_without_a_staging_step(self):
        """A router holds no data of its own, and the copy would only delay the run's first request."""
        command = _staged("staticWorkers", _router())

        assert command[:2] != ["bash", "-c"]
        assert "/scratch/dataset" not in " ".join(command)

    def test_only_a_trainer_is_asked_to_stage_anything(self):
        """The rule is per section, so a spec kind that ever slips past it would stage on every pod."""
        assert values_module._stages_inputs(_trainer())
        assert not values_module._stages_inputs(_engine())
        assert not values_module._stages_inputs(_router())

    def test_a_trainer_keeps_the_command_it_would_have_run(self):
        """Staging prefixes the launch; a rewritten command would start the wrong worker on the right data."""
        command = _staged("trainers", _trainer())

        assert command[2].endswith(" ".join(["--foo", "bar"]))

    def test_a_run_that_stages_nothing_leaves_every_command_alone(self):
        """Most runs read from the shared filesystem directly, and a bash wrapper would hide their exit codes."""
        command = values_module.build_values([_trainer()], LAYOUT)["run"]["trainers"][0]["command"]

        assert command[:2] != ["bash", "-c"]


class TestElasticCompatibility:
    def test_a_pool_that_started_with_one_cell_can_still_be_grown(self):
        """Elastic reads both sides of the diff, and an omitted count on the old side reads as a rewrite."""
        before = values_module.build_values([_engine(num_cells=1, gpus_per_engine=8)], LAYOUT)
        after = values_module.build_values([_engine(num_cells=4, gpus_per_engine=8)], LAYOUT)

        diff = elastic.diff_values(before, after)

        assert diff.is_allowed
        assert diff.scaled == ["run.inferenceEngines.[0].replicas: 1 -> 4"]

    def test_a_pool_can_be_shrunk_back_to_one_cell(self):
        """Scaling down to a single cell is how a run gives gpus back without being relaunched."""
        before = values_module.build_values([_trainer(num_cells=4, gpus_per_cell=8)], LAYOUT)
        after = values_module.build_values([_trainer(num_cells=1, gpus_per_cell=8)], LAYOUT)

        diff = elastic.diff_values(before, after)

        assert diff.is_allowed
        assert diff.scaled == ["run.trainers.[0].replicas: 4 -> 1"]

    def test_changing_how_many_static_workers_a_run_has_is_not_a_scaling(self):
        """A statefulset of session servers is part of what the run is, and elastic must not resize it silently."""
        before = values_module.build_values([_session_server(num_cells=1)], LAYOUT)
        after = values_module.build_values([_session_server(num_cells=2)], LAYOUT)

        assert not elastic.diff_values(before, after).is_allowed


class TestServedWorkerBootstrap:
    def test_names_the_spec_the_pod_has_to_rebuild(self):
        """A served pod constructs its worker from the spec, which its command is the only record of."""
        command = values_module.build_values([_trainer(num_cells=1, gpus_per_cell=1)], LAYOUT)["run"]["trainers"][0][
            "command"
        ]

        assert command[command.index("--pool-id") + 1] == "trainer-actor"

    def test_names_the_spec_table_the_pod_rebuilds_the_run_from(self):
        """The pod constructs its own worker, so it needs the same table the launcher rendered from."""
        command = values_module.build_values([_trainer(num_cells=1, gpus_per_cell=1)], LAYOUT)["run"]["trainers"][0][
            "command"
        ]

        assert command[command.index("--specs") + 1] == "miles.ray.specs.entrypoint.compute_specs_from_argv"

    def test_supervises_a_shared_pod_into_the_ranks_its_spec_packs_there(self):
        """The ranks of one pod run the same command, and only the supervisor knows how many to start."""
        command = values_module.build_values([_trainer(num_cells=1, gpus_per_cell=8)], LAYOUT)["run"]["trainers"][0][
            "command"
        ]

        assert command[command.index("--num-subprocesses") + 1] == "8"

    def test_keeps_the_entrypoint_flags_ahead_of_the_run_argv(self):
        """Everything after the separator belongs to the run, so a flag placed there would never be read."""
        command = values_module.build_values([_trainer(num_cells=1, gpus_per_cell=1)], LAYOUT)["run"]["trainers"][0][
            "command"
        ]

        assert command.index("--specs") < command.index("--")


class TestObjectNames:
    def test_every_pool_entry_carries_the_object_name_the_chart_must_render(self):
        """The chart computes no names, so an entry without one renders an object called nothing."""
        built = values_module.build_values([_router(), _engine(), _trainer()], LAYOUT)["run"]

        assert built["staticWorkers"][0]["objectName"] == "r-miles-run-inference-router-0"
        assert built["inferenceEngines"][0]["objectName"] == "r-miles-run-inference-engine-0-0"
        assert built["trainers"][0]["objectName"] == "r-miles-run-trainer-actor"

    def test_the_fixed_components_are_named_whether_or_not_they_are_enabled(self):
        """A schema-required field the launcher only sometimes writes would refuse half the runs."""
        assert values_module.build_values([], LAYOUT)["run"]["objectNames"] == {
            "orchestrator": "r-miles-run-orchestrator",
            "mooncakeMaster": "r-miles-run-mooncake-master",
            "colocatePairing": "r-miles-run-colocate-pairing",
        }

    def test_relaunching_the_same_run_writes_byte_identical_values(self):
        """helm upgrade replaces an object in place only while its name is unchanged."""
        specs = [_router(), _engine(), _trainer()]

        first = yaml.safe_dump(values_module.build_values(specs, LAYOUT), sort_keys=True)
        second = yaml.safe_dump(values_module.build_values(specs, LAYOUT), sort_keys=True)

        assert first == second


class TestDecidedAddresses:
    def test_a_command_reading_another_pool_gets_the_final_host_and_port(self):
        """Nothing expands a placeholder inside a container command, so the address is baked in."""
        built = values_module.build_values([_session_server(num_cells=1), _session_client()], LAYOUT)["run"]

        assert built["staticWorkers"][1]["command"][-1] == (
            "http://r-miles-run-session-server-0.r-miles-run-session-server:8000"
        )

    def test_only_a_statically_addressed_pool_is_in_the_address_book(self):
        """A pool's pods are named by its workload controller, so the launcher decides no address for it."""
        specs = [_router(), _engine(), _trainer()]

        assert sorted(values_module._decide_addresses(specs, "r")) == ["inference-router-0"]


class TestRanksPerPod:
    def test_the_supervisor_starts_the_ranks_the_provider_will_look_for(self):
        """The provider fans a pod out into the ranks its own command started, so the two cannot drift."""
        spec = make_trainer_spec(num_workers_per_cell=24, num_gpus_per_node=8)

        assert _launched_ranks_per_pod(spec) == spec.scheduling.ranks_per_pod()

    def test_an_engine_pod_runs_one_command_and_therefore_holds_one_rank(self):
        """An engine pod is a single server process spanning its gpus, whatever its spec counts as a worker."""
        spec = make_engine_spec()

        assert _launched_ranks_per_pod(spec) == 1
        assert spec.scheduling.ranks_per_pod() == 1


class TestPodsPerCell:
    def test_the_chart_deploys_the_pods_the_provider_expects_to_observe(self):
        """A cell rendered into fewer pods than the provider counts would never look complete."""
        spec = make_trainer_spec(num_workers_per_cell=24, num_gpus_per_node=8)

        assert _rendered_pods_per_cell(spec) == spec.scheduling.pods_per_cell()

    def test_a_cell_of_one_pod_is_rendered_without_a_size(self):
        """The chart's default is a single pod, and both sides have to read that absence the same way."""
        spec = make_engine_spec()

        assert _rendered_pods_per_cell(spec) == spec.scheduling.pods_per_cell() == 1


def _launched_ranks_per_pod(spec: BaseWorkerSpec) -> int:
    command = _rendered_entry(spec)["command"]
    if "--num-subprocesses" not in command:
        return 1
    return int(command[command.index("--num-subprocesses") + 1])


def _rendered_pods_per_cell(spec: BaseWorkerSpec) -> int:
    return _rendered_entry(spec).get("size", 1)


def _rendered_entry(spec: BaseWorkerSpec) -> dict:
    values = values_module.build_values(
        [spec],
        values_module.RunLayout(
            run_id="260101-000000-000",
            release=RELEASE,
            orchestrator_command=["python", "train.py"],
            worker_argv=[],
        ),
    )
    return next(
        entry
        for section in ("trainers", "inferenceEngines", "staticWorkers")
        for entry in values["run"][section]
        if entry["poolId"] == spec.name
    )
