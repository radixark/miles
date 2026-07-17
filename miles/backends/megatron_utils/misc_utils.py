def strip_param_name_prefix(name: str | None) -> str | None:
    if name is None:
        return None
    prefix = "module."
    while name.startswith(prefix):
        name = name.removeprefix(prefix)
    return name
