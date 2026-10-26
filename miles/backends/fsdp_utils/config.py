from typing import Any, ClassVar, Literal

from miles.utils.args.enhanced_argparse_namespace import EnhancedArgparseNamespace


class FsdpArgsNamespace(EnhancedArgparseNamespace):
    backend_name: Literal["fsdp"]
    _mutable_fields: ClassVar[frozenset[str]] = frozenset({"lr_decay_iters", "rank", "train_iters", "world_size"})

    def __init__(self, *, backend_name: Literal["fsdp"] = "fsdp", **values: Any) -> None:
        assert backend_name == "fsdp", f"Invalid FSDP backend name: {backend_name!r}"
        super().__init__(backend_name=backend_name, **values)
