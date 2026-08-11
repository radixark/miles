"""Guards the plumbing `start_weight_versions` has to survive to be usable.

The field is written in more than one place and declared in more than one wire
table, and every one of those omissions fails *silently*: the value is simply
absent downstream, so the staleness metric reads empty rather than raising. A
blank number, not a stack trace, is what makes this worth a test.

`weight_versions` is the reference throughout: it is the same per-call list, so
wherever it is handled, its submit-time counterpart has to be handled too.
"""

import inspect
import re

from miles.ray.rollout import train_data_conversion
from miles.rollout.generate_utils import sample_utils
from miles.rollout.session.samples import codec
from miles.utils.types import Sample

REFERENCE = "weight_versions"
FIELD = "start_weight_versions"


def _sample(**kwargs):
    return Sample(index=0, **kwargs)


class TestWireTableRegistration:
    """A field absent from these tables crosses the ray boundary as nothing."""

    def test_registered_in_train_data_spec(self):
        spec = train_data_conversion.ROLLOUT_DATA_VALUE_SPEC
        assert (FIELD in spec) == (REFERENCE in spec)

    def test_registered_in_samples_spec(self):
        spec = codec.SAMPLES_VALUE_SPEC
        assert (FIELD in spec) == (REFERENCE in spec)

    def test_v2_spec_inherits_it(self):
        # V2 derives from V1 today; if that ever becomes a separate literal, this
        # is where the omission would otherwise go unnoticed.
        assert (FIELD in codec.SAMPLES_VALUE_SPEC_V2) == (
            REFERENCE in codec.SAMPLES_VALUE_SPEC_V2
        )


class TestMultiTurnMergeCarriesIt:
    """Multi-turn merges concatenate the per-call lists. Handling only the old
    list would keep the last turn's version and silently drop the earlier ones,
    which is exactly the value the staleness `min` depends on."""

    def test_merge_source_handles_both_lists(self):
        source = inspect.getsource(sample_utils)
        # Negative lookbehind: "start_weight_versions" contains "weight_versions".
        old = len(re.findall(rf"(?<!start_){REFERENCE}", source))
        new = source.count(FIELD)
        assert old == new, (
            f"sample_utils handles {REFERENCE} {old} times and {FIELD} {new} "
            "times; a merge path is probably carrying one and not the other"
        )


class TestOldestStartWeightVersion:
    def test_min_across_calls(self):
        assert _sample(start_weight_versions=[9, 4, 7]).oldest_start_weight_version == 4

    def test_none_when_never_stamped(self):
        # Undeclared must stay undeclared: the buffer reads None as "cannot be
        # evaluated", not as version 0.
        assert _sample().oldest_start_weight_version is None

    def test_cleared_for_retry(self):
        sample = _sample(weight_versions=["5"], start_weight_versions=[5])
        sample.reset_for_retry()
        assert sample.start_weight_versions == []
        assert sample.weight_versions == []
