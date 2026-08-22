"""Guards the plumbing `start_weight_version` has to survive to be usable.

The field is declared in more than one wire table, and every omission fails
*silently*: the value is simply absent downstream, so the staleness metric reads
empty rather than raising. A blank number, not a stack trace, is what makes this
worth a test.

`weight_versions` is the reference throughout: it is the same per-call fact from
the other end of the request, so wherever it is handled, its submit-time
counterpart has to be handled too.
"""

import inspect

from miles.ray.rollout import train_data_conversion
from miles.rollout.generate_utils import sample_utils
from miles.rollout.session.samples import codec
from miles.utils.types import Sample

REFERENCE = "weight_versions"
FIELD = "start_weight_version"


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
    """Multi-turn merges concatenate the per-call lists but this field is one
    value for the whole trajectory, so the merge has to pick the older of the
    two. Keeping the later turn's version would understate staleness by exactly
    the span the field exists to measure."""

    def test_merge_keeps_the_earlier_start(self):
        merge = sample_utils._merge_start_weight_version
        assert merge(_sample(start_weight_version=9), _sample(start_weight_version=4)) == 4
        assert merge(_sample(start_weight_version=4), _sample(start_weight_version=9)) == 4

    def test_merge_ignores_an_undeclared_side(self):
        merge = sample_utils._merge_start_weight_version
        assert merge(_sample(start_weight_version=4), _sample()) == 4
        assert merge(_sample(), _sample(start_weight_version=4)) == 4
        assert merge(_sample(), _sample()) is None

    def test_merge_is_wired_into_the_sample_merge(self):
        # The helper is only a guard if the merge actually calls it.
        source = inspect.getsource(sample_utils._merge_sample_pair)
        assert f"{FIELD}=_merge_start_weight_version(a, b)" in source


class TestFieldContract:
    def test_none_when_never_stamped(self):
        # Undeclared must stay undeclared: the buffer reads None as "cannot be
        # evaluated", not as version 0.
        assert _sample().start_weight_version is None

    def test_cleared_for_retry(self):
        sample = _sample(weight_versions=["5"], start_weight_version=5)
        sample.reset_for_retry()
        assert sample.start_weight_version is None
        assert sample.weight_versions == []
