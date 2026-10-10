from dataclasses import dataclass, field


@dataclass(frozen=True)
class TokenTurn:
    id: str
    input_ids: tuple[int, ...]
    output_ids: tuple[int, ...]
    logprobs: tuple[float, ...]
    stop_reason: str


@dataclass
class TokenTrace:
    """The actual sampler inputs and outputs; never reconstruct tokens from messages."""

    turns: list[TokenTurn] = field(default_factory=list)

    def record(
        self, request_id: str, input_ids: list[int], tokens: list[int], logprobs: list[float], stop_reason: str
    ) -> TokenTurn:
        turn = TokenTurn(
            id=request_id,
            input_ids=tuple(input_ids),
            output_ids=tuple(tokens),
            logprobs=tuple(logprobs),
            stop_reason=stop_reason,
        )
        assert len(turn.output_ids) == len(turn.logprobs), "each sampled token must have a logprob"
        self.turns.append(turn)
        return turn
