import pytest
from tests.fast.backends.megatron_utils.backend_scope_fixtures import _tokenizer_builder, _tokenizer_initialization

from miles.backends.megatron_utils import initialize


class TestTokenizerMutationScope:
    @pytest.mark.parametrize("fails", [False, True])
    def test_tokenizer_sdk_can_publish_vocab_size_without_retaining_mutation_authority(
        self, monkeypatch: pytest.MonkeyPatch, fails: bool
    ) -> None:
        """Tokenizer SDK mutation remains scoped even when initialization raises."""
        args = _tokenizer_initialization(monkeypatch)
        monkeypatch.setattr(initialize, "_build_tokenizer", _tokenizer_builder(fails=fails))

        if fails:
            with pytest.raises(RuntimeError, match="tokenizer construction failed"):
                initialize.init(args)
        else:
            initialize.init(args)

        assert args.backend.padded_vocab_size == 128
        with pytest.raises(AttributeError, match="immutable"):
            args.backend.padded_vocab_size = 256
