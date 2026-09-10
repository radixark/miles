from tests.fast.import_isolation_utils import modules_imported_by


class TestLossHubImportWeight:
    def test_losses_import_does_not_require_ray(self):
        """The loss hub resolves custom losses through function_registry, so importing it must not pull in ray."""
        modules = modules_imported_by("miles.backends.training_utils.loss_hub.losses")

        assert "ray" not in modules
        assert "miles.utils.misc" not in modules
