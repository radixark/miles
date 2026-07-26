"""Verify the installed TransformerEngine wheel triplet."""

import importlib.metadata as metadata
import importlib.util
import sys


VERSION = "2.17.0"


def verify(core_dist: str) -> None:
    core = core_dist.replace("_", "-")
    expected = {
        "transformer-engine": VERSION,
        core: VERSION,
        "transformer-engine-torch": VERSION,
    }
    actual = {name: metadata.version(name) for name in expected}
    requires = {
        requirement.lower().replace("_", "-").replace(" ", "")
        for requirement in (metadata.requires("transformer-engine-torch") or [])
    }

    assert actual == expected, f"unexpected TransformerEngine versions: {actual}"
    assert f"{core}=={VERSION}" in requires, f"unexpected TransformerEngine torch requirements: {requires}"
    assert importlib.util.find_spec("transformer_engine") is not None, "transformer_engine package not found"


if __name__ == "__main__":
    verify(sys.argv[1])
