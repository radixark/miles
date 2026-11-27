import inspect

import pydantic
import yaml

from miles.utils.environ import use_legacy_rollout_v1
from miles.utils.file_arg_utils import resolve_file_arg
from miles.utils.function_registry import load_function
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.types import ClusterBackend, DeployComponent, WorkerCommBackend


class RewardPoolConfig(FrozenStrictBaseModel):
    """A pool of reward workers: `num_workers` cells of one worker each, every worker running one `scorer_path`.

    A reward function reaches the pool as its `<name>_pool` keyword argument; a GPU pool's workers each take one
    bundle of the run's placement group, after the rollout and eval GPUs.
    """

    name: str
    scorer_path: str
    scorer_kwargs: dict[str, object] = {}
    num_workers: int = pydantic.Field(gt=0)
    uses_gpu: bool
    batch_size: int = pydantic.Field(gt=0)
    # How long a call waits for concurrent calls to fill its batch; 0 sends each call as it comes.
    max_wait_ms: float = pydantic.Field(default=0.0, ge=0)
    num_gpus_per_node: int = pydantic.Field(gt=0)

    @pydantic.field_validator("name")
    @classmethod
    def _name_is_a_keyword(cls, name: str) -> str:
        assert name.isidentifier(), f"reward pool name {name!r} must be a Python identifier: it names `{name}_pool`"
        return name


class RewardPoolScalingConfig(FrozenStrictBaseModel):
    num_gpus: int = pydantic.Field(ge=0)
    gpu_offset: int = pydantic.Field(ge=0)


class RewardScalingConfig(FrozenStrictBaseModel):
    pools: dict[str, RewardPoolScalingConfig] = {}

    @property
    def num_gpus(self) -> int:
        return sum(pool.num_gpus for pool in self.pools.values())

    @classmethod
    def resolve(cls, pools: list[RewardPoolConfig], *, gpu_offset: int) -> "RewardScalingConfig":
        """Each GPU pool's slots follow the ones before it in the rollout view, as server groups' do."""
        resolved = {}
        for pool in pools:
            num_gpus = pool.num_workers if pool.uses_gpu else 0
            resolved[pool.name] = RewardPoolScalingConfig(num_gpus=num_gpus, gpu_offset=gpu_offset)
            gpu_offset += num_gpus
        return cls(pools=resolved)


def compute_reward_pools(args) -> tuple[list[RewardPoolConfig], RewardScalingConfig]:
    """The run's reward pools from `--reward-pool-config`, with GPU slots after the rollout and eval GPUs."""
    pools = _read_reward_pool_config(args)
    scaling = RewardScalingConfig.resolve(pools, gpu_offset=(args.rollout_num_gpus or 0) + args.eval_num_gpus)
    return pools, scaling


def _read_reward_pool_config(args) -> list[RewardPoolConfig]:
    if args.reward_pool_config is None:
        return []
    _check_run_can_host_reward_pools(args)
    entries = yaml.safe_load(resolve_file_arg(args.reward_pool_config)) or []
    assert isinstance(entries, list), f"--reward-pool-config must hold a list of pools, got {type(entries).__name__}"
    for entry in entries:
        assert "num_gpus_per_node" not in entry, "--reward-pool-config sets num_gpus_per_node, which Miles derives"
    pools = [RewardPoolConfig(**entry, num_gpus_per_node=args.num_gpus_per_node) for entry in entries]
    names = [pool.name for pool in pools]
    assert len(names) == len(set(names)), f"reward pool names must be unique, got {names}"
    return pools


def _check_run_can_host_reward_pools(args) -> None:
    problems = []
    if (
        args.cluster_backend != ClusterBackend.RAY.value
        or args.worker_comm_backend == WorkerCommBackend.RPC.value
        or args.deploy_component != DeployComponent.ALL.value
    ):
        # The pool's workers and their client live in one Ray deployment with the rollout executor.
        problems.append("one Ray deployment (--cluster-backend ray, a Ray comm backend, --deploy-component all)")
    if args.colocate:
        problems.append("GPUs of their own (no --colocate)")
    if args.rollout_external or args.debug_rollout_only or args.debug_train_only:
        problems.append("a rollout placement group (no external engines, --debug-rollout-only or --debug-train-only)")
    if use_legacy_rollout_v1():
        # The v1 rollout calls the reward function without the pools.
        problems.append("the v2 rollout (MILES_USE_LEGACY_ROLLOUT_V1 unset)")
    if plain := _plain_rollout_functions(args):
        problems.append(f"class-based rollout functions, which take the pools (not the plain functions {plain})")
    if problems:
        raise ValueError(f"--reward-pool-config needs {'; '.join(problems)}")


def _plain_rollout_functions(args) -> list[str]:
    """Rollout and eval functions that are not BaseRolloutFn classes: their adapter calls them without the pools."""
    # Local: miles.rollout imports the argument configs this module belongs to.
    from miles.rollout.base_types import BaseRolloutFn

    paths = {config.path for config in (args.rollout_function_path, args.eval_function_path) if config is not None}
    return sorted(
        path for path in paths if not (inspect.isclass(fn := load_function(path)) and issubclass(fn, BaseRolloutFn))
    )
