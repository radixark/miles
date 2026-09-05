from copy import copy, deepcopy
from dataclasses import FrozenInstanceError
from itertools import permutations, product

import pytest

from miles.utils.arg_resolution import (
    MISSING,
    ArgBatch,
    ArgResolutionContract,
    ArgResolutionError,
    Binding,
    ClaimPolicy,
    NonePolicy,
    PrimaryField,
    PrimarySchema,
    ResolvedArgs,
    SourceSpec,
    UnknownInputPolicy,
)


def _single_contract(*, binding=None, primary=None, schema_validate=None, unknown_inputs=UnknownInputPolicy.REJECT):
    return ArgResolutionContract(
        schema=PrimarySchema([primary or PrimaryField("value")], validate=schema_validate),
        sources=[SourceSpec("source", 0, [binding or Binding("input", "value")], unknown_inputs=unknown_inputs)],
    )


@pytest.mark.parametrize("name", ["", None, 1, False])
@pytest.mark.parametrize(
    "factory",
    [
        PrimaryField,
        lambda name: Binding(name, "value"),
        lambda name: Binding("input", name),
        lambda name: SourceSpec(name, 0, []),
    ],
)
def test_static_names_must_be_nonempty_strings(factory, name):
    with pytest.raises(ValueError, match="nonempty string"):
        factory(name)


@pytest.mark.parametrize("priority", [True, False, "1", 1.5, None])
def test_source_priority_requires_integer(priority):
    with pytest.raises(ValueError, match="integer"):
        SourceSpec("source", priority, [])


@pytest.mark.parametrize("priority", [True, False, "1", 1.5])
def test_binding_priority_requires_integer(priority):
    with pytest.raises(ValueError, match="integer"):
        Binding("input", "value", priority=priority)


@pytest.mark.parametrize(
    "factory, message",
    [
        (lambda: PrimaryField("value", normalize=True), "normalize"),
        (lambda: PrimaryField("value", validate=0), "validate"),
        (lambda: PrimarySchema([], validate="callable"), "validate"),
        (lambda: Binding("input", "value", project={}), "project"),
        (lambda: Binding("input", "value", policy="REGULAR"), "ClaimPolicy"),
        (lambda: Binding("input", "value", none_policy="VALUE"), "NonePolicy"),
        (lambda: SourceSpec("source", 0, [], unknown_inputs="REJECT"), "UnknownInputPolicy"),
        (lambda: PrimarySchema(None), "iterable"),
        (lambda: PrimarySchema(["value"]), "PrimaryField"),
        (lambda: SourceSpec("source", 0, None), "iterable"),
        (lambda: SourceSpec("source", 0, ["binding"]), "Binding"),
        (lambda: ArgResolutionContract(None, []), "PrimarySchema"),
        (lambda: ArgResolutionContract(PrimarySchema([]), None), "iterable"),
        (lambda: ArgResolutionContract(PrimarySchema([]), ["source"]), "SourceSpec"),
        (lambda: PrimarySchema([PrimaryField("value"), PrimaryField("value")]), "Duplicate primary"),
        (
            lambda: SourceSpec("source", 0, [Binding("input", "value"), Binding("input", "value", priority=1)]),
            "Duplicate bindings",
        ),
        (
            lambda: ArgResolutionContract(
                PrimarySchema([]), [SourceSpec("source", 0, []), SourceSpec("source", 1, [])]
            ),
            "Duplicate source",
        ),
        (
            lambda: ArgResolutionContract(PrimarySchema([]), [SourceSpec("source", 0, [Binding("input", "value")])]),
            "Unknown target",
        ),
    ],
)
def test_static_definition_errors(factory, message):
    with pytest.raises(ValueError, match=message) as caught:
        factory()
    assert not isinstance(caught.value, ArgResolutionError)


def test_sparse_output_and_baseline_has_ordinary_priority():
    contract = ArgResolutionContract(
        schema=PrimarySchema([PrimaryField("value"), PrimaryField("unclaimed")]),
        sources=[
            SourceSpec("baseline", -10, [Binding("value", "value")]),
            SourceSpec("request", 0, [Binding("alias", "value")]),
            SourceSpec("absent", 100, [Binding("input", "value")]),
        ],
    )
    result = contract.resolve([ArgBatch("request", {"alias": 2}), ArgBatch("baseline", {"value": 1})])
    assert result.values == {"value": 2}
    assert list(result.provenance) == ["value"]
    assert [(claim.source_id, claim.selected) for claim in result.provenance["value"]] == [
        ("baseline", False),
        ("request", True),
    ]
    assert contract.resolve([]).values == {}
    assert contract.resolve([]).provenance == {}


def test_binding_priority_override_and_fan_in_out():
    contract = ArgResolutionContract(
        schema=PrimarySchema([PrimaryField("left", normalize=int), PrimaryField("right")]),
        sources=[
            SourceSpec(
                "low",
                -2,
                [Binding("both", "left", priority=5), Binding("both", "right", project=lambda value: [value])],
            ),
            SourceSpec("high", 2, [Binding("alias", "left")]),
        ],
    )
    result = contract.resolve([ArgBatch("low", {"both": "7"}), ArgBatch("high", {"alias": 3})])
    assert result.values == {"left": 7, "right": ["7"]}
    assert [(claim.source_id, claim.priority, claim.value, claim.selected) for claim in result.provenance["left"]] == [
        ("high", 2, 3, False),
        ("low", 5, 7, True),
    ]
    assert result.provenance["right"][0].input_name == "both"
    assert result.provenance["right"][0].target_name == "right"
    assert result.provenance["right"][0].policy is ClaimPolicy.REGULAR


@pytest.mark.parametrize("policy", list(ClaimPolicy))
def test_missing_never_claims_or_runs_callbacks(policy):
    def unexpected(value):
        pytest.fail("A missing input reached a callback")

    contract = _single_contract(
        binding=Binding("input", "value", policy=policy, none_policy=NonePolicy.REJECT, project=unexpected),
        primary=PrimaryField("value", normalize=unexpected, validate=unexpected),
    )
    for values in ({}, {"input": MISSING}):
        result = contract.resolve([ArgBatch("source", values)])
        assert result.values == result.provenance == {}
    assert copy(MISSING) is MISSING
    assert deepcopy(MISSING) is MISSING


@pytest.mark.parametrize("none_policy", list(NonePolicy))
def test_none_policies(none_policy):
    seen = []
    contract = _single_contract(
        binding=Binding("input", "value", none_policy=none_policy, project=lambda value: seen.append(value) or value)
    )
    if none_policy is NonePolicy.REJECT:
        with pytest.raises(ArgResolutionError, match="rejects None"):
            contract.resolve([ArgBatch("source", {"input": None})])
        assert seen == []
    else:
        result = contract.resolve([ArgBatch("source", {"input": None})])
        assert result.values == ({"value": None} if none_policy is NonePolicy.VALUE else {})
        assert seen == ([None] if none_policy is NonePolicy.VALUE else [])


@pytest.mark.parametrize("none_policy", list(NonePolicy))
@pytest.mark.parametrize("value", [None, 0, False, "", []])
def test_forbidden_rejects_before_none_policy_or_projection(none_policy, value):
    contract = _single_contract(
        binding=Binding(
            "input", "value", policy=ClaimPolicy.FORBIDDEN, none_policy=none_policy, project=lambda value: MISSING
        )
    )
    with pytest.raises(ArgResolutionError, match="forbidden"):
        contract.resolve([ArgBatch("source", {"input": value})])


def test_projection_can_suppress_claim_but_normalization_cannot():
    def unexpected(value):
        pytest.fail("A suppressed input reached normalization")

    projected = _single_contract(
        binding=Binding("input", "value", project=lambda value: MISSING),
        primary=PrimaryField("value", normalize=unexpected),
    )
    assert projected.resolve([ArgBatch("source", {"input": 4})]).values == {}
    normalized = _single_contract(primary=PrimaryField("value", normalize=lambda value: MISSING))
    with pytest.raises(ArgResolutionError, match="normalization returned MISSING"):
        normalized.resolve([ArgBatch("source", {"input": 4})])


def test_project_then_normalize_then_validate_every_claim_including_losers():
    validated = []

    def validate(value):
        validated.append(value)
        if value < 0:
            raise ValueError("negative claim")

    contract = ArgResolutionContract(
        schema=PrimarySchema([PrimaryField("value", normalize=int, validate=validate)]),
        sources=[
            SourceSpec("low", 0, [Binding("input", "value", project=lambda value: value["inner"])]),
            SourceSpec("winner", 1, [Binding("input", "value")]),
        ],
    )
    result = contract.resolve([ArgBatch("low", {"input": {"inner": "3"}}), ArgBatch("winner", {"input": "4"})])
    assert result.values == {"value": 4}
    assert validated == [3, 4]
    with pytest.raises(ArgResolutionError, match="low.*input.*value.*negative claim"):
        contract.resolve([ArgBatch("winner", {"input": "4"}), ArgBatch("low", {"input": {"inner": "-1"}})])


@pytest.mark.parametrize("policy", [ClaimPolicy.REGULAR, ClaimPolicy.REQUIRE_EQUAL])
def test_equal_top_priority_claims_are_all_selected(policy):
    contract = ArgResolutionContract(
        schema=PrimarySchema([PrimaryField("value", normalize=int)]),
        sources=[
            SourceSpec("b", 1, [Binding("input", "value", policy=policy)]),
            SourceSpec("a", 1, [Binding("input", "value")]),
            SourceSpec("c", 0, [Binding("input", "value")]),
        ],
    )
    result = contract.resolve(
        [ArgBatch("b", {"input": "2"}), ArgBatch("c", {"input": 2}), ArgBatch("a", {"input": 2})]
    )
    assert result.values == {"value": 2}
    assert [(claim.source_id, claim.selected) for claim in result.provenance["value"]] == [
        ("a", True),
        ("b", True),
        ("c", False),
    ]


@pytest.mark.parametrize("required_source", [None, "low", "high"])
def test_requirement_equality_includes_lower_priority_claims(required_source):
    contract = ArgResolutionContract(
        schema=PrimarySchema([PrimaryField("value")]),
        sources=[
            SourceSpec(
                name,
                priority,
                [
                    Binding(
                        "input",
                        "value",
                        policy=ClaimPolicy.REQUIRE_EQUAL if name == required_source else ClaimPolicy.REGULAR,
                    )
                ],
            )
            for name, priority in [("low", 0), ("high", 1)]
        ],
    )
    batches = [ArgBatch("low", {"input": 1}), ArgBatch("high", {"input": 2})]
    if required_source is None:
        assert contract.resolve(batches).values == {"value": 2}
    else:
        with pytest.raises(ArgResolutionError, match="Conflicting claims.*value"):
            contract.resolve(batches)


def test_unequal_top_priority_tie_fails():
    contract = ArgResolutionContract(
        schema=PrimarySchema([PrimaryField("value")]),
        sources=[SourceSpec("source", 0, [Binding("one", "value"), Binding("two", "value")])],
    )
    for values in ({"one": 1, "two": 2}, {"two": 2, "one": 1}):
        with pytest.raises(ArgResolutionError, match="Conflicting claims"):
            contract.resolve([ArgBatch("source", values)])


def test_registration_batch_binding_and_input_order_do_not_change_provenance():
    expected = None
    bindings = [Binding("z", "value"), Binding("a", "value"), Binding("z", "other")]
    for binding_order, source_order, batch_order, input_order, field_order in product(
        permutations(bindings),
        permutations(["z", "a"]),
        permutations(["z", "a"]),
        permutations(["z", "a"]),
        permutations(["value", "other"]),
    ):
        contract = ArgResolutionContract(
            schema=PrimarySchema([PrimaryField(name) for name in field_order]),
            sources=[SourceSpec(name, 1, binding_order) for name in source_order],
        )
        result = contract.resolve([ArgBatch(name, dict.fromkeys(input_order, 4)) for name in batch_order])
        if expected is None:
            expected = result
        assert result == expected
        assert list(result.values) == list(result.provenance) == ["other", "value"]
        assert [(claim.source_id, claim.input_name) for claim in result.provenance["value"]] == [
            ("a", "a"),
            ("a", "z"),
            ("z", "a"),
            ("z", "z"),
        ]
        assert all(claim.selected for claims in result.provenance.values() for claim in claims)


@pytest.mark.parametrize("value", [None, False, 1, "1", [1], {"nested": 1}])
def test_raw_values_have_no_implicit_coercion(value):
    result = _single_contract().resolve([ArgBatch("source", {"input": value})])
    assert result.values["value"] is value


@pytest.mark.parametrize("unknown_policy", list(UnknownInputPolicy))
def test_unknown_input_policy(unknown_policy):
    contract = _single_contract(unknown_inputs=unknown_policy)
    batch = ArgBatch("source", {"input": 3, "extra": MISSING})
    if unknown_policy is UnknownInputPolicy.REJECT:
        with pytest.raises(ArgResolutionError, match="Unknown input 'extra'"):
            contract.resolve([batch])
    else:
        result = contract.resolve([batch])
        assert result.values == {"value": 3}
        assert list(result.provenance) == ["value"]
        assert [claim.input_name for claim in result.provenance["value"]] == ["input"]


@pytest.mark.parametrize(
    "factory, message",
    [
        (lambda: ArgBatch("", {}), "nonempty string"),
        (lambda: ArgBatch(None, {}), "nonempty string"),
        (lambda: ArgBatch("source", []), "mapping"),
        (lambda: ArgBatch("source", None), "mapping"),
        (lambda: _single_contract().resolve(None), "iterable"),
        (lambda: _single_contract().resolve([{}]), "ArgBatch"),
        (lambda: _single_contract().resolve([ArgBatch("unknown", {})]), "Unknown batch"),
        (
            lambda: _single_contract().resolve([ArgBatch("source", {}), ArgBatch("source", {})]),
            "Duplicate batch",
        ),
    ],
)
def test_bad_runtime_batches_raise_resolution_error(factory, message):
    with pytest.raises(ArgResolutionError, match=message):
        factory()


@pytest.mark.parametrize("stage", ["projection", "normalization", "validation", "schema"])
def test_callback_errors_have_context_and_original_cause(stage):
    cause = RuntimeError("extension failed")

    def fail(value):
        raise cause

    contract = _single_contract(
        binding=Binding("input", "value", project=fail if stage == "projection" else None),
        primary=PrimaryField(
            "value",
            normalize=fail if stage == "normalization" else None,
            validate=fail if stage == "validation" else None,
        ),
        schema_validate=fail if stage == "schema" else None,
    )
    with pytest.raises(ArgResolutionError) as caught:
        contract.resolve([ArgBatch("source", {"input": 3})])
    assert caught.value.__cause__ is cause
    assert "extension failed" in str(caught.value)
    if stage == "schema":
        assert "Schema validation" in str(caught.value)
    else:
        assert f"Source 'source' input 'input' target 'value' {stage}" in str(caught.value)


@pytest.mark.parametrize("stage", ["projection", "normalization", "validation", "schema"])
def test_callback_base_exception_propagates(stage):
    def interrupted(value):
        raise KeyboardInterrupt("stop")

    contract = _single_contract(
        binding=Binding("input", "value", project=interrupted if stage == "projection" else None),
        primary=PrimaryField(
            "value",
            normalize=interrupted if stage == "normalization" else None,
            validate=interrupted if stage == "validation" else None,
        ),
        schema_validate=interrupted if stage == "schema" else None,
    )
    with pytest.raises(KeyboardInterrupt, match="stop"):
        contract.resolve([ArgBatch("source", {"input": 3})])


@pytest.mark.parametrize("fail_in_bool", [False, True])
def test_equality_failures_are_contextual(fail_in_bool):
    cause = RuntimeError("cannot compare")

    class BadEquality:
        def __eq__(self, other):
            if fail_in_bool:
                return self
            raise cause

        def __bool__(self):
            raise cause

    contract = ArgResolutionContract(
        schema=PrimarySchema([PrimaryField("value")]),
        sources=[SourceSpec("source", 0, [Binding("a", "value"), Binding("b", "value")])],
    )
    with pytest.raises(ArgResolutionError, match="Equality comparison.*value") as caught:
        contract.resolve([ArgBatch("source", {"a": BadEquality(), "b": BadEquality()})])
    assert caught.value.__cause__ is cause


def test_definition_batch_and_result_collections_are_shallow_immutable_snapshots():
    fields = [PrimaryField("value")]
    bindings = [Binding("input", "value")]
    source = SourceSpec("source", 0, bindings)
    sources = [source]
    schema = PrimarySchema(fields)
    contract = ArgResolutionContract(schema, sources)
    nested = []
    values = {"input": nested}
    batch = ArgBatch("source", values)
    fields.clear()
    bindings.clear()
    sources.clear()
    values["input"] = "changed"
    result = contract.resolve([batch])
    assert result.values["value"] is nested
    assert batch.values["input"] is nested
    assert len(schema.fields) == len(source.bindings) == len(contract.sources) == 1
    assert isinstance(result.provenance["value"], tuple)
    for mapping in [batch.values, result.values, result.provenance]:
        with pytest.raises(TypeError):
            mapping["new"] = 1
    for instance, name in [
        (source, "priority"),
        (batch, "source_id"),
        (result, "values"),
        (result.provenance["value"][0], "selected"),
    ]:
        with pytest.raises(FrozenInstanceError):
            setattr(instance, name, None)
    repeated = contract.resolve([batch])
    assert repeated == result
    assert repeated.values is not result.values
    assert repeated.provenance is not result.provenance
    output_values = {"value": 1}
    output_claims = {"value": list(result.provenance["value"])}
    snapshot = ResolvedArgs(output_values, output_claims)
    output_values.clear()
    output_claims["value"].clear()
    assert snapshot.values == {"value": 1}
    assert len(snapshot.provenance["value"]) == 1


def test_cross_field_validation_is_read_only_sparse_and_atomic():
    observed = []

    def validate(values):
        assert "unclaimed" not in values
        with pytest.raises(TypeError):
            values["injected"] = True
        observed.append(values)
        if values["min"] > values["max"]:
            raise ValueError("min exceeds max")

    contract = ArgResolutionContract(
        schema=PrimarySchema([PrimaryField(name) for name in ["min", "max", "unclaimed"]], validate=validate),
        sources=[SourceSpec("source", 0, [Binding(name, name) for name in ["min", "max"]])],
    )
    valid = ArgBatch("source", {"min": 1, "max": 2})
    previous = contract.resolve([valid])
    bad_values = {"min": 3, "max": 2}
    with pytest.raises(ArgResolutionError, match="Schema validation.*min exceeds max"):
        contract.resolve([ArgBatch("source", bad_values)])
    assert bad_values == {"min": 3, "max": 2}
    assert previous.values == valid.values == {"min": 1, "max": 2}
    assert contract.resolve([valid]) == previous
    assert observed[0] == observed[2] == previous.values
    assert observed[1] == bad_values
