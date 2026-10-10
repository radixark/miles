from collections.abc import Sequence

from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.types import StopReason, Trajectory, Transition

import tinker
from miles.tinker.core.token_trace import TokenTurn


def turns_to_trajectory(turns: Sequence[TokenTurn]) -> Trajectory:
    if not turns:
        raise ValueError("the session recorded no samples")
    transitions = [
        Transition(
            ob=tinker.ModelInput.from_ints(list(turn.input_ids)),
            ac=TokensWithLogprobs(
                tokens=list(turn.output_ids),
                maybe_logprobs=list(turn.logprobs),
                stop_reason=turn.stop_reason,
            ),
            reward=0.0,
            episode_done=index == len(turns) - 1,
        )
        for index, turn in enumerate(turns)
    ]
    last = turns[-1]
    stop_reason = {"stop": StopReason.COMPLETED, "length": StopReason.MAX_TOKENS}[last.stop_reason]
    return Trajectory(
        transitions=transitions,
        final_ob=tinker.ModelInput.from_ints([*last.input_ids, *last.output_ids]),
        stop_reason=str(stop_reason),
    )
