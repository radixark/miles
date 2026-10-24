from miles.utils.arguments import reset_arg
import argparse
import pytest

class TestResetArgument:

    def test_reset_changes_default_without_duplicating_an_existing_option(self) -> None:
        """Reset preserves explicit user values while replacing the fallback."""
        parser = argparse.ArgumentParser()
        parser.add_argument('--count', type=int, default=1, choices=[1, 2])
        reset_arg(parser=parser, name='--count', type=int, default=2, choices=[1, 2])
        assert parser.parse_args([]).count == 2
        assert parser.parse_args(['--count', '1']).count == 1

    @pytest.mark.parametrize('kwargs', [{'type': str}, {'type': int, 'choices': [3]}, {'action': 'store_true'}])
    def test_incompatible_resets_leave_the_original_default_intact(self, kwargs: dict) -> None:
        """Rejected declarations must not partially change the existing option."""
        parser = argparse.ArgumentParser()
        parser.add_argument('--count', type=int, default=1, choices=[1, 2])
        with pytest.raises(AssertionError, match='Cannot reset'):
            reset_arg(parser=parser, name='--count', default=9, **kwargs)
        assert parser.parse_args([]).count == 1

    def test_reset_registers_a_missing_option(self) -> None:
        """Backend-specific absent options are registered with the requested contract."""
        parser = argparse.ArgumentParser()
        reset_arg(parser=parser, name='--count', type=int, default=2)
        assert parser.parse_args(['--count', '8']).count == 8
