from types import SimpleNamespace

from miles.backends.megatron_utils.arguments import _normalize_gloo_process_groups_arg


class TestNormalizeGlooProcessGroupsArg:
    def test_the_old_megatron_name_populates_the_current_contract(self):
        """The old Megatron flag name remains usable by current Miles consumers."""
        args = SimpleNamespace(enable_gloo_process_groups=True)

        _normalize_gloo_process_groups_arg(args)

        assert args.use_gloo_process_groups is True

    def test_the_current_megatron_name_takes_precedence(self):
        """The current Megatron flag is never overwritten by a legacy alias."""
        args = SimpleNamespace(use_gloo_process_groups=False, enable_gloo_process_groups=True)

        _normalize_gloo_process_groups_arg(args)

        assert args.use_gloo_process_groups is False
