"""Grade an Inkling multiple-choice answer by the text the model showed, not by its thinking.

Inkling writes thinking blocks before the one text block that carries the answer letter. The gpqa
grader reads a flat response and would pick "option is (B)" out of the thinking, so this reward
parses the response tokens with the Inkling response parser and grades only the visible content.
"""

from functools import cache

from miles.rollout.rm_hub.gpqa import compute_gpqa_reward
from miles.utils.chat_template_utils.inkling_parser import InklingResponseParser
from miles.utils.processing_utils import load_tokenizer
from miles.utils.types import Sample


@cache
def _parser(hf_checkpoint: str, chat_template_path: str | None) -> InklingResponseParser:
    tokenizer = load_tokenizer(hf_checkpoint, chat_template_path=chat_template_path, trust_remote_code=True)
    return InklingResponseParser(tokenizer)


def visible_content(parser: InklingResponseParser, sample: Sample) -> str:
    response_tokens = sample.tokens[len(sample.tokens) - sample.response_length :]
    return parser.parse(response_tokens, finish_reason=None).client_message["content"]


async def reward_func(args, sample: Sample, **kwargs) -> float:
    parser = _parser(args.hf_checkpoint, args.chat_template_path)
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return compute_gpqa_reward(visible_content(parser, sample), sample.label, metadata=metadata)
