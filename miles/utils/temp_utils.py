import dataclasses
import os
from typing import List

ENABLE_DEBUG_PRINT = True

ENABLE_DEBUG_SHORT_PROMPT = True
# ENABLE_DEBUG_SHORT_PROMPT = False

# ENABLE_DEBUG_PROFILE = True
ENABLE_DEBUG_PROFILE = False

if bool(int(os.environ.get("SGLANG_DUMPER_ENABLE", "1"))):
    print("when dumper is enabled, cannot do profile")
    ENABLE_DEBUG_PROFILE = False


PROFILE_OUTPUT_DIR = "/host_home/temp_sglang_server2local"


@dataclasses.dataclass
class TokenIdsAndLogprobs:
    token_ids: List[int]
    logprobs: List[float]

    def __add__(self, other):
        return TokenIdsAndLogprobs(
            token_ids=self.token_ids + other.token_ids,
            logprobs=self.logprobs + other.logprobs,
        )

    @classmethod
    def compare(cls, a: "TokenIdsAndLogprobs", b: "TokenIdsAndLogprobs"):
        assert len(a.token_ids) == len(b.token_ids)
        token_match = a.token_ids == b.token_ids
        logprobs_match = a.logprobs == b.logprobs

        if token_match:
            print(f"Token match: {a.token_ids}")
        else:
            print(f"❗Token mismatch: {a.token_ids=} {b.token_ids=}")

        if logprobs_match:
            print(f"Logprobs match:", a.logprobs)
        else:
            print(f"❗Logprobs mismatch")
            print(
                "    A:   ",
                [f"{x:.10f}" if x is not None else "None" for x in a.logprobs],
            )
            print(
                "    B:   ",
                [f"{x:.10f}" if x is not None else "None" for x in b.logprobs],
            )
            diff = [abs(x - y) if x is not None else float("nan") for x, y in zip(a.logprobs, b.logprobs)]
            print("    Diff:", [f"{x:.10e}" for x in diff])

        return token_match and logprobs_match


def extract_ids_and_logprobs(responses):
    def _extract_part(response, name):
        token_ids, logprobs = [], []
        for item in response["meta_info"][name]:
            logprob, token_id, text = item
            token_ids.append(token_id)
            logprobs.append(logprob)
        return TokenIdsAndLogprobs(token_ids=token_ids, logprobs=logprobs)

    def _extract_one_response(response):
        input = _extract_part(response, "input_token_logprobs")
        output = _extract_part(response, "output_token_logprobs")
        return dict(input=input, output=output, io=input + output)

    if not isinstance(responses, list):
        responses = [responses]
    return [_extract_one_response(x) for x in responses]
