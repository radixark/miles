from tests.fast.launch_scripts.sh_harness import REPO_ROOT

_HARDCODED_CHECKOUTS = ("/root/miles", "/workspace/miles")


def test_no_shell_script_hardcodes_the_checkout_location():
    """A script that assumes one absolute checkout only runs inside one container image."""
    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for root in (REPO_ROOT / "scripts", REPO_ROOT / "examples")
        for path in root.rglob("*.sh")
        for text in [path.read_text(errors="replace")]
        if any(hardcoded in text for hardcoded in _HARDCODED_CHECKOUTS)
    ]

    assert offenders == []
