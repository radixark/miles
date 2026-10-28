import pytest

from miles.utils.audit_utils.config_snapshot.enabled import SNAPSHOT_TESTING_ENABLED

if not SNAPSHOT_TESTING_ENABLED:
    pytest.skip("Snapshot testing is disabled by op48-277", allow_module_level=True)

from collections.abc import Callable


from miles.utils.audit_utils.config_snapshot.normalizer import normalize_record


class TestBackendRankNormalization:
    def test_training_backend_rank_is_normalized_without_mutating_input_or_nested_user_values(
        self, make_record: Callable
    ) -> None:
        """Only the declared trainer backend rank receives the rank placeholder."""
        record = make_record(config={"backend": {"rank": 3, "custom": {"rank": 4}}})

        normalized = normalize_record(record)

        assert normalized["args"]["backend"] == {"rank": "$RANK", "custom": {"rank": 4}}
        assert record.config["args"]["backend"]["rank"] == 3

    @pytest.mark.parametrize("rank", [-1, True, 1.5, "0", None])
    def test_invalid_backend_rank_is_rejected_instead_of_hidden_by_normalization(
        self, make_record: Callable, rank: object
    ) -> None:
        """Backend ranks must be nonnegative integers rather than coercible or missing values."""
        with pytest.raises(AssertionError, match="Unexpected args.backend.rank"):
            normalize_record(make_record(config={"backend": {"rank": rank}}))

    @pytest.mark.parametrize("backend", [None, [], "megatron"])
    def test_training_backend_must_be_a_snapshot_object(self, make_record: Callable, backend: object) -> None:
        """A malformed backend container cannot silently bypass training snapshot normalization."""
        with pytest.raises(AssertionError, match="config.args.backend object"):
            normalize_record(make_record(config={"backend": backend}))
