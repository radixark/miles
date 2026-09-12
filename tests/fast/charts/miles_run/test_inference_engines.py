import json

from tests.fast.charts.utils import (
    objects_of_kind,
    render_run,
    requires_helm,
    single_object_of_kind,
    with_object_names,
)

ONE_GROUP = [
    {
        "name": "inference-engine-0-0",
        "replicas": 2,
        "size": 4,
        "command": [
            "python",
            "-m",
            "sglang.launch_server",
            "--node-rank",
            "$(LWS_WORKER_INDEX)",
            "--dist-init-addr",
            "$(LWS_LEADER_ADDRESS):9000",
        ],
        "env": {"NVSHMEM_DISABLE_NCCL": "1"},
        "ports": [{"name": "primary", "port": 8000}],
        "resources": {"limits": {"nvidia.com/gpu": 8}},
    }
]


def _render(engines=ONE_GROUP, *args):
    return render_run("--set-json", f"run.inferenceEngines={json.dumps(with_object_names(engines))}", *args)


def _leader_worker_set(engines=ONE_GROUP):
    return single_object_of_kind(_render(engines), "LeaderWorkerSet")


@requires_helm
class TestInferenceEngines:
    def test_a_group_is_one_engine_and_a_pod_is_one_of_its_nodes(self):
        """A multi-node engine must be scheduled and restarted as a unit, which is what a group gives."""
        spec = _leader_worker_set()["spec"]

        assert spec["replicas"] == 2
        assert spec["leaderWorkerTemplate"]["size"] == 4

    def test_restarts_the_whole_engine_when_one_rank_dies(self):
        """A surviving rank of a dead engine would keep serving requests it can no longer answer."""
        assert _leader_worker_set()["spec"]["leaderWorkerTemplate"]["restartPolicy"] == "RecreateGroupOnPodRestart"

    def test_serves_every_rank_from_one_template(self):
        """Ranks differ only by index, and kubelet expands $(LWS_WORKER_INDEX) per pod."""
        template = _leader_worker_set()["spec"]["leaderWorkerTemplate"]

        assert "leaderTemplate" not in template
        assert "$(LWS_WORKER_INDEX)" in template["workerTemplate"]["spec"]["containers"][0]["command"]

    def test_passes_the_command_through_unchanged(self):
        """The launcher computed the sglang argv; the chart must not reinterpret its placeholders."""
        container = _leader_worker_set()["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]["containers"][0]

        assert container["command"] == ONE_GROUP[0]["command"]

    def test_requests_the_gpus_the_launcher_computed(self):
        """An engine pod owning a whole node must hold every gpu on it."""
        container = _leader_worker_set()["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]["containers"][0]

        assert container["resources"] == {"limits": {"nvidia.com/gpu": 8}}

    def test_disables_the_service_link_environment(self):
        """Engine pods import miles too, so the injected <SERVICE>_PORT vars would break arg parsing."""
        pod = _leader_worker_set()["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]

        assert pod["enableServiceLinks"] is False

    def test_defaults_a_single_pod_engine(self):
        """A one-node engine is the common case and must not have to spell out replicas and size."""
        spec = _leader_worker_set([{"name": "engine", "command": ["python"]}])["spec"]

        assert (spec["replicas"], spec["leaderWorkerTemplate"]["size"]) == (1, 1)

    def test_renders_nothing_for_an_external_inference_pool(self):
        """External rollout brings its own engines, and the run must install without any."""
        assert objects_of_kind(render_run(), "LeaderWorkerSet") == []

    def test_gives_every_group_of_every_model_its_own_pool(self):
        """Multi-model and prefill/decode runs differ only in how many groups the specs name."""
        groups = [
            {"name": "inference-engine-0-0", "command": ["python", "-m", "a"]},
            {"name": "inference-engine-1-0", "command": ["python", "-m", "b"]},
            {"name": "inference-engine-1-1", "command": ["python", "-m", "b", "--disaggregation-mode", "decode"]},
        ]

        rendered = objects_of_kind(_render(groups), "LeaderWorkerSet")

        assert [obj["metadata"]["name"] for obj in rendered] == [
            "myrun-miles-run-inference-engine-0-0",
            "myrun-miles-run-inference-engine-1-0",
            "myrun-miles-run-inference-engine-1-1",
        ]

    def test_sizes_each_group_on_its_own(self):
        """A prefill engine and a decode engine of one model rarely span the same number of nodes."""
        groups = [
            {"name": "prefill", "replicas": 1, "size": 4, "command": ["python"]},
            {"name": "decode", "replicas": 8, "size": 1, "command": ["python"]},
        ]

        rendered = objects_of_kind(_render(groups), "LeaderWorkerSet")
        sizes = {
            obj["metadata"]["name"]: (obj["spec"]["replicas"], obj["spec"]["leaderWorkerTemplate"]["size"])
            for obj in rendered
        }

        assert sizes == {"myrun-miles-run-prefill": (1, 4), "myrun-miles-run-decode": (8, 1)}
