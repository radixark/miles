class AdapterSourceError(ValueError):
    """A source the caller named is not one this run can load: unrecognized, malformed, or another parametrization.

    Raised by the library's source checks only, so a caller can tell a refused source from a storage fault or a bug.
    """
