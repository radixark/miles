import shlex

from examples.infra_features.split_deployment.address_book import INIT_EXPECTED_NUM_CELLS_FLAG

from miles.ray.specs.inference import INFERENCE_CONTROLLER_ADDR_FLAG
from miles.ray.specs.train import TRAINER_CONTROLLER_ADDRS_FLAG
from miles.utils.external_utils.command_utils.common import MOONCAKE_INIT_KWARGS_FLAG, ArgvManipulator

ROLLOUT_NUM_GPUS_FLAG: str = "--rollout-num-gpus"

FLAGS_A_COMMAND_OF_ONE_SPLIT_RUN_MAY_DIFFER_ON: tuple[str, ...] = (
    INFERENCE_CONTROLLER_ADDR_FLAG,
    INIT_EXPECTED_NUM_CELLS_FLAG,
    TRAINER_CONTROLLER_ADDRS_FLAG,
    ROLLOUT_NUM_GPUS_FLAG,
)

FLAGS_A_SPLIT_RUN_MAY_DIFFER_FROM_AN_UNSPLIT_ONE_ON: tuple[str, ...] = (
    *FLAGS_A_COMMAND_OF_ONE_SPLIT_RUN_MAY_DIFFER_ON,
    MOONCAKE_INIT_KWARGS_FLAG,
)


def value_of(train_args: str, flag: str) -> str:
    values = ArgvManipulator.get(shlex.split(train_args), flag)
    assert len(values) == 1, f"{flag} is declared {len(values)} time(s) in these arguments"
    return values[0]


def values_after(train_args: str, flag: str) -> list[str]:
    tokens = shlex.split(train_args)
    kept: list[str] = []
    for token in tokens[tokens.index(flag) + 1 :]:
        if token.startswith("--"):
            break
        kept.append(token)
    return kept


def shared_argv(train_args: str, *, differing_flags: tuple[str, ...]) -> list[str]:
    kept: list[str] = []
    skipping = False
    for token in shlex.split(train_args):
        if token in differing_flags:
            skipping = True
        elif token.startswith("--"):
            skipping = False
            kept.append(token)
        elif not skipping:
            kept.append(token)
    return kept
