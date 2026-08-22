import shlex

import pytest
from examples.multi_policy.run_solver_verifier_gsm8k import (
    LEADER_MODEL_ID,
    MODEL_IDS,
    SHARED_TRAINER_OVERRIDES,
    ScriptArgs,
    build_train_args,
    compute_events_dir,
    compute_megatron_config,
    compute_sglang_config,
    compute_trainer_id,
)
from tests.e2e.conftest_multi_policy import NUM_VERIFIED_ARGS_PER_POLICY

from miles.utils.file_arg_utils import PSEUDO_FILE_PREFIX

RUN_ID: str = "demo"
CONFIG_FLAGS: tuple[str, ...] = ("--megatron-config", "--sglang-config")


@pytest.fixture(autouse=True)
def _no_wandb(monkeypatch) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)


@pytest.fixture
def args() -> ScriptArgs:
    return ScriptArgs(run_id=RUN_ID)


@pytest.fixture
def flags(args: ScriptArgs) -> dict[str, str | None]:
    return _flags_of(build_train_args(args))


class TestBuildTrainArgs:
    def test_the_recipe_is_exactly_these_flags_and_values(self, flags, args):
        """A flag silently dropped from an 80-line recipe surfaces as an 8-gpu run that hangs, not as a red test."""
        solver_path = args.model_path_of_model_id[LEADER_MODEL_ID]

        assert {flag: value for flag, value in flags.items() if flag not in CONFIG_FLAGS} == {
            "--hf-checkpoint": f"{solver_path}/",
            "--ref-load": f"{solver_path}/",
            "--custom-generate-function-path": "examples.multi_policy.solver_verifier.generate",
            "--fully-async": None,
            "--prompt-data": f"{args.data_dir}/gsm8k/train.parquet",
            "--input-key": "messages",
            "--label-key": "label",
            "--rollout-shuffle": None,
            "--num-rollout": str(args.num_rollout),
            "--rollout-batch-size": "8",
            "--n-samples-per-prompt": "4",
            "--rollout-max-response-len": "1024",
            "--rollout-temperature": "0.8",
            "--global-batch-size": "32",
            "--pause-generation-mode": "in_place",
            "--optimizer": "adam",
            "--lr": "1e-6",
            "--advantage-estimator": "grpo",
            "--use-kl-loss": None,
            "--tensor-model-parallel-size": "1",
            "--sequence-parallel": None,
            "--pipeline-model-parallel-size": "1",
            "--context-parallel-size": "1",
            "--expert-model-parallel-size": "1",
            "--expert-tensor-parallel-size": "1",
            "--use-dynamic-batch-size": None,
            "--max-tokens-per-gpu": "9216",
            "--rollout-num-gpus-per-engine": "1",
            "--sglang-mem-fraction-static": "0.65",
            "--sglang-enable-metrics": None,
            "--ci-test": None,
            "--save-debug-event-data": str(compute_events_dir(args)),
            "--attention-dropout": "0.0",
            "--hidden-dropout": "0.0",
            "--accumulate-allreduce-grads-in-fp32": None,
            "--attention-softmax-in-fp32": None,
            "--attention-backend": "flash",
            "--actor-num-nodes": "1",
            "--actor-num-gpus-per-node": str(args.actor_num_gpus_per_policy),
            "--rollout-num-gpus": str(args.rollout_num_gpus),
            "--megatron-to-hf-mode": "bridge",
        }

    def test_the_run_generates_without_retracting_what_it_has_in_flight(self, flags):
        """Retracting, the default, can deadlock flush_cache under a fully async load: the run hangs, silently."""
        assert flags["--fully-async"] is None
        assert flags["--pause-generation-mode"] == "in_place"

    def test_the_per_policy_configs_are_handed_over_as_files_of_their_own(self, flags):
        """Their contents are asserted below; here they only have to reach the run at all."""
        assert sorted(flag for flag in flags if flag in CONFIG_FLAGS) == sorted(CONFIG_FLAGS)
        assert all(flags[flag].startswith(PSEUDO_FILE_PREFIX) for flag in CONFIG_FLAGS)

    def test_a_run_told_how_many_engine_gpus_to_expect_declares_that_instead(self, args):
        """A split deployment installs the engines of one policy, and counts only those."""
        flags = _flags_of(build_train_args(args, rollout_num_gpus=1))

        assert flags["--rollout-num-gpus"] == "1"


class TestComputeMegatronConfig:
    def test_the_run_trains_exactly_the_policies_the_example_declares(self, args):
        """A trainer short of the model ids is a policy nothing ever trains, and the run still starts."""
        trainers = compute_megatron_config(args)["trainers"]

        assert [one["model_id"] for one in trainers] == MODEL_IDS
        assert [one["trainer_id"] for one in trainers] == [compute_trainer_id(one) for one in MODEL_IDS]

    def test_every_policy_is_given_the_arguments_the_e2e_test_counts(self, args):
        """The e2e assertion verifies a fixed number of overrides per policy, and would shrink with this one."""
        trainers = compute_megatron_config(args)["trainers"]

        assert {one["model_id"]: len(one["overrides"]) for one in trainers} == NUM_VERIFIED_ARGS_PER_POLICY

    def test_every_policy_is_given_the_shared_hyperparameters_of_this_recipe(self, args):
        """A policy trained on other hyperparameters than its neighbour is not the comparison this run makes."""
        for trainer in compute_megatron_config(args)["trainers"]:
            assert {key: trainer["overrides"][key] for key in SHARED_TRAINER_OVERRIDES} == SHARED_TRAINER_OVERRIDES

    def test_a_run_of_one_policy_carries_only_that_policy(self, args):
        """A split deployment installs a trainer per policy, and hands each command its own config."""
        trainers = compute_megatron_config(args, model_ids=[LEADER_MODEL_ID])["trainers"]

        assert [one["model_id"] for one in trainers] == [LEADER_MODEL_ID]


class TestComputeSglangConfig:
    def test_the_run_serves_exactly_the_policies_the_example_declares(self, args):
        """A policy without engines generates nothing, and only its own reward would ever show it."""
        models = compute_sglang_config(args)["sglang"]

        assert [one["name"] for one in models] == MODEL_IDS
        assert [one["model_path"] for one in models] == [args.model_path_of_model_id[one] for one in MODEL_IDS]

    def test_every_policy_is_served_by_the_engine_gpus_the_run_declares(self, args):
        """Engines short of the gpus the run counts leave it waiting for cells nothing installs."""
        models = compute_sglang_config(args)["sglang"]
        declared = [group["num_gpus"] for one in models for group in one["server_groups"]]

        assert declared == [args.rollout_num_gpus_per_model] * len(MODEL_IDS)
        assert sum(declared) == args.rollout_num_gpus

    def test_a_run_of_one_policy_serves_only_that_policy(self, args):
        """A split deployment installs the engines of one policy, and hands that command its own config."""
        models = compute_sglang_config(args, model_ids=[LEADER_MODEL_ID])["sglang"]

        assert [one["name"] for one in models] == [LEADER_MODEL_ID]


def _flags_of(train_args: str) -> dict[str, str | None]:
    tokens = shlex.split(train_args)
    flags: dict[str, str | None] = {}
    for index, token in enumerate(tokens):
        if not token.startswith("--"):
            continue

        assert token not in flags, f"{token} is declared twice in these arguments, and only one of them can win"
        following = tokens[index + 1] if index + 1 < len(tokens) else None
        flags[token] = None if following is None or following.startswith("--") else following
    return flags
