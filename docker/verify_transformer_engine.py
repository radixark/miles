"""Verify the installed TransformerEngine wheel triplet."""

import importlib.metadata as metadata
import importlib.util
import re
import sys


VERSION = "2.17.0"


def _dist_name(requirement: str) -> str:
    """'onnxscript', 'torch>=2.1', 'foo; extra == \"bar\"' -> normalised dist name."""
    return re.split(r"[<>=!~;\[ ]", requirement.strip(), maxsplit=1)[0].lower().replace("_", "-")


def verify(core_dist: str) -> None:
    core = core_dist.replace("_", "-")
    expected = {
        "transformer-engine": VERSION,
        core: VERSION,
        "transformer-engine-torch": VERSION,
    }
    actual = {name: metadata.version(name) for name in expected}
    raw_requires = metadata.requires("transformer-engine-torch") or []
    requires = {requirement.lower().replace("_", "-").replace(" ", "") for requirement in raw_requires}

    assert actual == expected, f"unexpected TransformerEngine versions: {actual}"
    assert f"{core}=={VERSION}" in requires, f"unexpected TransformerEngine torch requirements: {requires}"
    assert importlib.util.find_spec("transformer_engine") is not None, "transformer_engine package not found"

    # The triplet is installed with --no-deps, so nothing makes transformer_engine_torch's
    # own runtime deps present. Missing ones do not surface until something imports
    # transformer_engine.pytorch, which is far too late -- a missing onnxscript shipped
    # a green image that failed every GPU test.
    missing = []
    for requirement in raw_requires:
        name = _dist_name(requirement)
        if name.startswith("transformer-engine"):
            continue  # pinned above
        try:
            metadata.version(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
    assert not missing, f"transformer_engine_torch runtime deps not installed: {missing}"


if __name__ == "__main__":
    verify(sys.argv[1])
