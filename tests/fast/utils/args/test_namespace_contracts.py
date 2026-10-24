from typing import ClassVar
import pytest
from miles.utils.args.enhanced_argparse_namespace import EnhancedArgparseNamespace


class _Namespace(EnhancedArgparseNamespace):
    _mutable_fields: ClassVar[frozenset[str]] = frozenset({'step'})


class TestNamespaceMutation:

    def test_nested_scopes_restore_immutability_after_an_exception(self) -> None:
        """An exceptional inner scope must preserve the outer lease and finally revoke it."""
        config = _Namespace(step=1, model='fixed')
        with config.mutable():
            with pytest.raises(RuntimeError):
                with config.mutable():
                    config.step = 2
                    raise RuntimeError('abort')
            config.step = 3
            with pytest.raises(AttributeError, match='immutable'):
                config.model = 'changed'
        assert config.step == 3
        with pytest.raises(AttributeError, match='immutable'):
            config.step = 4

    def test_deletion_obeys_the_same_scope_and_allowlist_as_assignment(self) -> None:
        """Deletion cannot bypass the configuration mutation boundary."""
        config = _Namespace(step=1, model='fixed')
        with pytest.raises(AttributeError, match='immutable'):
            del config.step
        with config.mutable():
            with pytest.raises(AttributeError, match='immutable'):
                del config.model
            del config.step
            config.step = 5
        assert vars(config) == {'step': 5, 'model': 'fixed'}
