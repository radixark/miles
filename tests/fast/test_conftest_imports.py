from tests.fast.import_isolation_utils import modules_imported_by


def test_root_conftest_does_not_import_rollout_dependencies():
    modules = modules_imported_by("tests.conftest")

    unexpected_modules = modules & {
        "tests.fast.fixtures.generation_fixtures",
        "tests.fast.fixtures.rollout_fixtures",
        "torch",
        "ray",
        "sglang",
    }
    assert not unexpected_modules
