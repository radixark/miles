from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from types import MappingProxyType
from typing import Any


class ArgResolutionError(ValueError):
    pass


class ClaimPolicy(Enum):
    REGULAR = auto()
    REQUIRE_EQUAL = auto()
    FORBIDDEN = auto()


class NonePolicy(Enum):
    VALUE = auto()
    MISSING = auto()
    REJECT = auto()


class UnknownInputPolicy(Enum):
    REJECT = auto()
    IGNORE = auto()


class _Missing(Enum):
    TOKEN = auto()

    def __repr__(self):
        return "MISSING"


MISSING = _Missing.TOKEN


def _check_name(value, context):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a nonempty string")


def _check_priority(value, context):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer, not a boolean")


def _check_callback(value, context):
    if value is not None and not callable(value):
        raise ValueError(f"{context} must be callable or None")


def _snapshot_definitions(values, expected_type, context):
    try:
        snapshot = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{context} must be iterable") from exc
    if any(not isinstance(value, expected_type) for value in snapshot):
        raise ValueError(f"{context} must contain {expected_type.__name__} instances")
    return snapshot


@dataclass(frozen=True)
class PrimaryField:
    name: str
    normalize: Callable[[Any], Any] | None = None
    validate: Callable[[Any], None] | None = None

    def __post_init__(self):
        _check_name(self.name, "Primary field name")
        _check_callback(self.normalize, f"Field {self.name!r} normalize")
        _check_callback(self.validate, f"Field {self.name!r} validate")


@dataclass(frozen=True)
class PrimarySchema:
    fields: tuple[PrimaryField, ...]
    validate: Callable[[Mapping[str, Any]], None] | None = None

    def __post_init__(self):
        fields = _snapshot_definitions(self.fields, PrimaryField, "Schema fields")
        if len({item.name for item in fields}) != len(fields):
            raise ValueError("Duplicate primary field names")
        _check_callback(self.validate, "Schema validate")
        object.__setattr__(self, "fields", fields)


@dataclass(frozen=True)
class Binding:
    input_name: str
    target_name: str
    priority: int | None = None
    policy: ClaimPolicy = ClaimPolicy.REGULAR
    none_policy: NonePolicy = NonePolicy.VALUE
    project: Callable[[Any], Any] | None = None

    def __post_init__(self):
        _check_name(self.input_name, "Binding input name")
        _check_name(self.target_name, "Binding target name")
        if self.priority is not None:
            _check_priority(self.priority, "Binding priority")
        if not isinstance(self.policy, ClaimPolicy):
            raise ValueError("Binding policy must be a ClaimPolicy")
        if not isinstance(self.none_policy, NonePolicy):
            raise ValueError("Binding none_policy must be a NonePolicy")
        _check_callback(self.project, "Binding project")


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    priority: int
    bindings: tuple[Binding, ...]
    unknown_inputs: UnknownInputPolicy = UnknownInputPolicy.REJECT

    def __post_init__(self):
        _check_name(self.source_id, "Source ID")
        _check_priority(self.priority, "Source priority")
        bindings = _snapshot_definitions(self.bindings, Binding, "Source bindings")
        if len({(item.input_name, item.target_name) for item in bindings}) != len(bindings):
            raise ValueError(f"Duplicate bindings for source {self.source_id!r}")
        if not isinstance(self.unknown_inputs, UnknownInputPolicy):
            raise ValueError("Source unknown_inputs must be an UnknownInputPolicy")
        object.__setattr__(self, "bindings", bindings)


@dataclass(frozen=True)
class ArgBatch:
    source_id: str
    values: Mapping[str, Any]

    def __post_init__(self):
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ArgResolutionError("Batch source ID must be a nonempty string")
        if not isinstance(self.values, Mapping):
            raise ArgResolutionError(f"Batch {self.source_id!r} values must be a mapping")
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))


@dataclass(frozen=True)
class Claim:
    source_id: str
    input_name: str
    target_name: str
    priority: int
    policy: ClaimPolicy
    value: Any
    selected: bool


@dataclass(frozen=True)
class ResolvedArgs:
    """Read-only top-level snapshots; nested values remain shared."""

    values: Mapping[str, Any]
    provenance: Mapping[str, tuple[Claim, ...]]

    def __post_init__(self):
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))
        provenance = {name: tuple(claims) for name, claims in self.provenance.items()}
        object.__setattr__(self, "provenance", MappingProxyType(provenance))


def _call(callback, value, context):
    try:
        return callback(value)
    except Exception as exc:
        raise ArgResolutionError(f"{context} failed: {exc}") from exc


def _equal(left, right, target_name):
    try:
        return bool(left == right)
    except Exception as exc:
        raise ArgResolutionError(f"Equality comparison for target {target_name!r} failed: {exc}") from exc


@dataclass(frozen=True)
class ArgResolutionContract:
    """Resolve sparse primary values; callbacks must be pure and validators reject by raising."""

    schema: PrimarySchema
    sources: tuple[SourceSpec, ...]
    _fields: Mapping[str, PrimaryField] = field(init=False, repr=False, compare=False)
    _sources: Mapping[str, SourceSpec] = field(init=False, repr=False, compare=False)
    _inputs: Mapping[str, frozenset[str]] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.schema, PrimarySchema):
            raise ValueError("Contract schema must be a PrimarySchema")
        sources = _snapshot_definitions(self.sources, SourceSpec, "Contract sources")
        fields = {item.name: item for item in self.schema.fields}
        source_map = {item.source_id: item for item in sources}
        if len(source_map) != len(sources):
            raise ValueError("Duplicate source IDs")
        for source in sources:
            for binding in source.bindings:
                if binding.target_name not in fields:
                    raise ValueError(f"Unknown target {binding.target_name!r} in source {source.source_id!r}")
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "_fields", MappingProxyType(fields))
        object.__setattr__(self, "_sources", MappingProxyType(source_map))
        inputs = {source.source_id: frozenset(item.input_name for item in source.bindings) for source in sources}
        object.__setattr__(self, "_inputs", MappingProxyType(inputs))

    def resolve(self, batches: Iterable[ArgBatch]) -> ResolvedArgs:
        try:
            batches = tuple(batches)
        except TypeError as exc:
            raise ArgResolutionError("Batches must be iterable") from exc
        seen = set()
        for batch in batches:
            if not isinstance(batch, ArgBatch):
                raise ArgResolutionError("Batches must contain ArgBatch instances")
            if batch.source_id in seen:
                raise ArgResolutionError(f"Duplicate batch source ID {batch.source_id!r}")
            seen.add(batch.source_id)
            if batch.source_id not in self._sources:
                raise ArgResolutionError(f"Unknown batch source ID {batch.source_id!r}")
            source = self._sources[batch.source_id]
            if source.unknown_inputs is UnknownInputPolicy.REJECT:
                for name in batch.values:
                    if name not in self._inputs[batch.source_id]:
                        raise ArgResolutionError(f"Unknown input {name!r} in source {batch.source_id!r}")

        claims_by_target = {}
        for batch in sorted(batches, key=lambda item: item.source_id):
            source = self._sources[batch.source_id]
            for binding in sorted(source.bindings, key=lambda item: (item.target_name, item.input_name)):
                claim = self._claim(source, binding, batch.values.get(binding.input_name, MISSING))
                if claim is not None:
                    claims_by_target.setdefault(claim.target_name, []).append(claim)

        values, provenance = {}, {}
        for name, claims in sorted(claims_by_target.items()):
            highest = max(claim.priority for claim in claims)
            winners = [claim for claim in claims if claim.priority == highest]
            if any(claim.policy is ClaimPolicy.REQUIRE_EQUAL for claim in claims):
                compared = claims
            else:
                compared = winners
            if any(not _equal(compared[0].value, claim.value, name) for claim in compared[1:]):
                raise ArgResolutionError(f"Conflicting claims for target {name!r}")
            values[name] = winners[0].value
            provenance[name] = tuple(replace(claim, selected=claim.priority == highest) for claim in claims)

        if self.schema.validate is not None:
            _call(self.schema.validate, MappingProxyType(values), "Schema validation")
        return ResolvedArgs(values=values, provenance=provenance)

    def _claim(self, source, binding, value):
        if value is MISSING:
            return None
        context = f"Source {source.source_id!r} input {binding.input_name!r} target {binding.target_name!r}"
        if binding.policy is ClaimPolicy.FORBIDDEN:
            raise ArgResolutionError(f"{context} is forbidden")
        if value is None:
            if binding.none_policy is NonePolicy.MISSING:
                return None
            if binding.none_policy is NonePolicy.REJECT:
                raise ArgResolutionError(f"{context} rejects None")
        if binding.project is not None:
            value = _call(binding.project, value, f"{context} projection")
            if value is MISSING:
                return None
        primary = self._fields[binding.target_name]
        if primary.normalize is not None:
            value = _call(primary.normalize, value, f"{context} normalization")
            if value is MISSING:
                raise ArgResolutionError(f"{context} normalization returned MISSING")
        if primary.validate is not None:
            _call(primary.validate, value, f"{context} validation")
        return Claim(
            source_id=source.source_id,
            input_name=binding.input_name,
            target_name=binding.target_name,
            priority=source.priority if binding.priority is None else binding.priority,
            policy=binding.policy,
            value=value,
            selected=False,
        )
