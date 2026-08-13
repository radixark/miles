import dataclasses
import datetime
import enum
import json
import math
import uuid
from decimal import Decimal
from pathlib import PurePosixPath
from typing import Any, Literal

import pytest
from pydantic import ValidationError
from pydantic_core import PydanticSerializationError
from typing_extensions import TypedDict

from miles.utils.pydantic_utils import StrictBaseModel
from miles.utils.workers.rpc.common.serialization import RpcSerializer


class Colour(enum.Enum):
    RED = "red"
    BLUE = "blue"


class Level(enum.IntEnum):
    LOW = 1
    HIGH = 2


class Inner(StrictBaseModel):
    name: str
    score: float


class Outer(StrictBaseModel):
    inner: Inner
    tags: list[str]
    lookup: dict[str, Inner]


@dataclasses.dataclass
class Point:
    x: int
    y: int


@dataclasses.dataclass
class Reading:
    value: float


class Blob(StrictBaseModel):
    data: bytes


class Options(TypedDict):
    retries: int
    label: str


NON_FINITE_FLOAT_TAG = "__miles_non_finite_float__"


_ROUNDTRIP_CASES = [
    ("int", int, 42),
    ("negative_int", int, -7),
    ("big_int", int, 2**70),
    ("float", float, 1.5),
    ("bool", bool, True),
    ("str", str, "text"),
    ("unicode_str", str, "中文 🚀 \\ \" '"),
    ("empty_str", str, ""),
    ("none", type(None), None),
    ("list_of_int", list[int], [1, 2, 3]),
    ("empty_list", list[int], []),
    ("nested_list", list[list[int]], [[1], [2, 3], []]),
    ("dict_str_int", dict[str, int], {"a": 1}),
    ("dict_of_lists", dict[str, list[int]], {"a": [1, 2]}),
    ("tuple", tuple[int, str], (1, "a")),
    ("variadic_tuple", tuple[int, ...], (1, 2, 3)),
    ("set", set[int], {1, 2, 3}),
    ("frozenset", frozenset[str], frozenset({"a", "b"})),
    ("optional_present", int | None, 5),
    ("optional_absent", int | None, None),
    ("union", int | str, "either"),
    ("literal", Literal["a", "b"], "b"),
    ("str_enum", Colour, Colour.RED),
    ("int_enum", Level, Level.HIGH),
    ("uuid", uuid.UUID, uuid.UUID("12345678-1234-5678-1234-567812345678")),
    ("datetime", datetime.datetime, datetime.datetime(2026, 7, 27, 12, 30, tzinfo=datetime.timezone.utc)),
    ("naive_datetime", datetime.datetime, datetime.datetime(2026, 7, 27, 12, 30)),
    ("date", datetime.date, datetime.date(2026, 7, 27)),
    ("time", datetime.time, datetime.time(12, 30, 15)),
    ("timedelta", datetime.timedelta, datetime.timedelta(seconds=90)),
    ("decimal", Decimal, Decimal("1.25")),
    ("path", PurePosixPath, PurePosixPath("/tmp/x")),
    ("bytes", bytes, b"raw-bytes"),
    ("model", Inner, Inner(name="x", score=1.5)),
    (
        "nested_model",
        Outer,
        Outer(inner=Inner(name="x", score=1.0), tags=["a"], lookup={"k": Inner(name="y", score=2.0)}),
    ),
    ("list_of_models", list[Inner], [Inner(name="a", score=1.0), Inner(name="b", score=2.0)]),
    ("dict_of_models", dict[str, Inner], {"k": Inner(name="a", score=1.0)}),
    ("optional_model", Inner | None, Inner(name="a", score=1.0)),
    ("dataclass", Point, Point(x=1, y=2)),
    ("typed_dict", Options, {"retries": 2, "label": "x"}),
    ("any_scalar", Any, 3),
    ("any_container", Any, {"k": [1, {"n": None}]}),
    ("deeply_nested", dict[str, list[dict[str, int]]], {"a": [{"b": 1}, {"c": 2}]}),
]


_NON_FINITE_CASES = [
    ("float_nan", float, math.nan),
    ("float_inf", float, math.inf),
    ("float_negative_inf", float, -math.inf),
    ("optional_float_nan", float | None, math.nan),
    ("optional_float_inf", float | None, math.inf),
    ("any_nan", Any, math.nan),
    ("any_negative_inf", Any, -math.inf),
    ("bare_dict_nan", dict, {"a": math.nan, "b": 1.0}),
    ("dict_str_float_inf", dict[str, float], {"a": math.inf}),
    ("list_of_float_nan", list[float], [math.nan, 1.0, -math.inf]),
    ("tuple_of_float_nan_and_inf", tuple[float, float], (math.nan, math.inf)),
    ("model_field_nan", Inner, Inner(name="x", score=math.nan)),
    ("model_field_negative_inf", Inner, Inner(name="x", score=-math.inf)),
    ("dataclass_field_inf", Reading, Reading(value=math.inf)),
    ("deeply_nested_nan", dict[str, list[dict[str, float]]], {"a": [{"b": math.nan}]}),
]

_BYTES_CASES = [
    ("non_utf8_bytes", bytes, b"\x00\x80\xff"),
    ("empty_bytes", bytes, b""),
    ("list_of_bytes", list[bytes], [b"\x00\x80\xff", b""]),
    ("dict_of_bytes", dict[str, bytes], {"k": b"\x00\x80\xff"}),
]


def _serializer(annotation: type) -> RpcSerializer:
    return RpcSerializer.create(
        query_model_name="Query", query_fields={"payload": (annotation, ...)}, result_annotation=annotation
    )


def _through_the_wire(payload: Any) -> Any:
    return json.loads(json.dumps(payload))


def _through_the_strict_wire(payload: Any) -> Any:
    return json.loads(json.dumps(payload, allow_nan=False))


def _equal_including_non_finite_floats(actual: Any, expected: Any) -> bool:
    return repr(actual) == repr(expected)


@pytest.mark.parametrize("case", _ROUNDTRIP_CASES, ids=[case[0] for case in _ROUNDTRIP_CASES])
class TestWireRoundtrip:
    def test_result_survives_the_wire_as_the_declared_type(self, case):
        """Every supported result type comes back from json as the declared python type."""
        _, annotation, value = case
        serializer = _serializer(annotation)
        assert serializer.decode_result(_through_the_wire(serializer.encode_result(value))) == value

    def test_argument_survives_the_wire_as_the_declared_type(self, case):
        """Every supported argument type reaches the worker as the declared python type."""
        _, annotation, value = case
        serializer = _serializer(annotation)
        assert (
            serializer.decode_query(_through_the_wire(serializer.encode_query({"payload": value})))["payload"] == value
        )


class TestTypeRevival:
    def test_model_result_is_revived_as_a_model(self):
        """A model result arrives as an instance rather than the raw json dict."""
        serializer = _serializer(Inner)
        revived = serializer.decode_result(_through_the_wire(serializer.encode_result(Inner(name="x", score=1.0))))
        assert isinstance(revived, Inner)

    def test_nested_model_argument_is_revived_all_the_way_down(self):
        """A nested model argument arrives with its inner models revived too."""
        value = Outer(inner=Inner(name="x", score=1.0), tags=[], lookup={"k": Inner(name="y", score=2.0)})
        serializer = _serializer(Outer)
        revived = serializer.decode_query(_through_the_wire(serializer.encode_query({"payload": value})))["payload"]
        assert isinstance(revived.lookup["k"], Inner)

    def test_dataclass_argument_is_revived_as_the_dataclass(self):
        """A dataclass argument is handed to the method as an instance, not a dict."""
        serializer = _serializer(Point)
        revived = serializer.decode_query(_through_the_wire(serializer.encode_query({"payload": Point(x=1, y=2)})))
        assert isinstance(revived["payload"], Point)

    def test_model_argument_is_revived_as_a_model(self):
        """A model argument is handed to the method as an instance, not a dict."""
        serializer = _serializer(Inner)
        revived = serializer.decode_query(
            _through_the_wire(serializer.encode_query({"payload": Inner(name="x", score=1.0)}))
        )
        assert isinstance(revived["payload"], Inner)

    def test_enum_result_is_revived_as_the_enum_member(self):
        """An enum result arrives as the member rather than its bare value."""
        serializer = _serializer(Colour)
        assert serializer.decode_result(_through_the_wire(serializer.encode_result(Colour.BLUE))) is Colour.BLUE

    def test_int_enum_result_is_revived_as_the_enum_member_not_a_bare_int(self):
        """An int enum result arrives as the member itself rather than the equal plain int."""
        serializer = _serializer(Level)
        revived = serializer.decode_result(_through_the_wire(serializer.encode_result(Level.HIGH)))
        assert type(revived) is Level
        assert revived is Level.HIGH

    def test_int_enum_argument_is_revived_as_the_enum_member_not_a_bare_int(self):
        """An int enum argument reaches the method as the member itself rather than the equal plain int."""
        serializer = _serializer(Level)
        revived = serializer.decode_query(_through_the_wire(serializer.encode_query({"payload": Level.LOW})))
        assert type(revived["payload"]) is Level
        assert revived["payload"] is Level.LOW

    def test_tuple_result_is_revived_as_a_tuple(self):
        """A tuple result arrives as a tuple even though json carries a list."""
        serializer = _serializer(tuple[int, str])
        assert isinstance(serializer.decode_result(_through_the_wire(serializer.encode_result((1, "a")))), tuple)

    def test_set_result_is_revived_as_a_set(self):
        """A set result arrives as a set even though json carries a list."""
        serializer = _serializer(set[int])
        assert isinstance(serializer.decode_result(_through_the_wire(serializer.encode_result({1, 2}))), set)

    def test_datetime_result_is_revived_as_a_datetime(self):
        """A datetime result arrives as a datetime rather than an iso string."""
        serializer = _serializer(datetime.datetime)
        value = datetime.datetime(2026, 7, 27, 12, 0, tzinfo=datetime.timezone.utc)
        revived = serializer.decode_result(_through_the_wire(serializer.encode_result(value)))
        assert isinstance(revived, datetime.datetime)
        assert revived.utcoffset() == value.utcoffset()

    def test_dataclass_result_is_revived_as_the_dataclass(self):
        """A dataclass result arrives as an instance rather than a dict."""
        serializer = _serializer(Point)
        assert isinstance(
            serializer.decode_result(_through_the_wire(serializer.encode_result(Point(x=1, y=2)))), Point
        )

    def test_decimal_result_keeps_its_precision(self):
        """A decimal result keeps full precision instead of degrading to a float."""
        serializer = _serializer(Decimal)
        value = Decimal("0.1234567890123456789")
        assert serializer.decode_result(_through_the_wire(serializer.encode_result(value))) == value

    def test_int_result_does_not_widen_to_float(self):
        """An int result stays an int across the wire."""
        serializer = _serializer(int)
        assert isinstance(serializer.decode_result(_through_the_wire(serializer.encode_result(1))), int)

    def test_bool_result_does_not_collapse_to_int(self):
        """A bool result stays a bool rather than becoming 0 or 1."""
        serializer = _serializer(bool)
        assert serializer.decode_result(_through_the_wire(serializer.encode_result(True))) is True

    def test_encoded_payloads_are_plain_json_types(self):
        """Encoding produces json-native values so the wire never sees python objects."""
        serializer = _serializer(Outer)
        value = Outer(inner=Inner(name="x", score=1.0), tags=["a"], lookup={})
        assert json.dumps(serializer.encode_result(value))


@pytest.mark.parametrize("case", _NON_FINITE_CASES, ids=[case[0] for case in _NON_FINITE_CASES])
class TestNonFiniteFloatRoundtrip:
    def test_non_finite_result_survives_the_wire(self, case):
        """Every non-finite float result comes back from json as the same non-finite float."""
        _, annotation, value = case
        serializer = _serializer(annotation)
        decoded = serializer.decode_result(_through_the_strict_wire(serializer.encode_result(value)))
        assert _equal_including_non_finite_floats(decoded, value)

    def test_non_finite_argument_survives_the_wire(self, case):
        """Every non-finite float argument reaches the worker as the same non-finite float."""
        _, annotation, value = case
        serializer = _serializer(annotation)
        decoded = serializer.decode_query(_through_the_strict_wire(serializer.encode_query({"payload": value})))
        assert _equal_including_non_finite_floats(decoded["payload"], value)

    def test_encoded_non_finite_result_is_strict_json(self, case):
        """Encoding a non-finite float result yields a payload json accepts without allow_nan."""
        _, annotation, value = case
        serializer = _serializer(annotation)
        assert json.dumps(serializer.encode_result(value), allow_nan=False)

    def test_encoded_non_finite_argument_is_strict_json(self, case):
        """Encoding a non-finite float argument yields a payload json accepts without allow_nan."""
        _, annotation, value = case
        serializer = _serializer(annotation)
        assert json.dumps(serializer.encode_query({"payload": value}), allow_nan=False)


class TestNonFiniteFloatValues:
    def test_nan_result_decodes_to_nan(self):
        """A NaN result decodes back to a real NaN float rather than None or a string."""
        serializer = _serializer(float)
        decoded = serializer.decode_result(_through_the_strict_wire(serializer.encode_result(float("nan"))))
        assert isinstance(decoded, float)
        assert math.isnan(decoded)

    def test_infinity_result_decodes_to_positive_infinity(self):
        """An infinity result decodes back to positive infinity rather than None or a string."""
        serializer = _serializer(float)
        decoded = serializer.decode_result(_through_the_strict_wire(serializer.encode_result(float("inf"))))
        assert decoded == math.inf

    def test_negative_infinity_result_decodes_to_negative_infinity(self):
        """A negative infinity result decodes back to negative infinity rather than None or a string."""
        serializer = _serializer(float)
        decoded = serializer.decode_result(_through_the_strict_wire(serializer.encode_result(float("-inf"))))
        assert decoded == -math.inf

    def test_nan_result_under_any_annotation_decodes_to_nan(self):
        """A NaN result declared as Any keeps its float identity instead of becoming a marker dict."""
        serializer = _serializer(Any)
        decoded = serializer.decode_result(_through_the_strict_wire(serializer.encode_result(math.nan)))
        assert isinstance(decoded, float)
        assert math.isnan(decoded)

    def test_nan_result_inside_a_model_field_decodes_to_nan(self):
        """A NaN nested in a model field decodes back to NaN on the revived model."""
        serializer = _serializer(Inner)
        decoded = serializer.decode_result(
            _through_the_strict_wire(serializer.encode_result(Inner(name="x", score=math.nan)))
        )
        assert isinstance(decoded, Inner)
        assert math.isnan(decoded.score)

    def test_infinity_result_inside_a_dataclass_field_decodes_to_infinity(self):
        """An infinity nested in a dataclass field decodes back to infinity on the revived dataclass."""
        serializer = _serializer(Reading)
        decoded = serializer.decode_result(_through_the_strict_wire(serializer.encode_result(Reading(value=math.inf))))
        assert isinstance(decoded, Reading)
        assert decoded.value == math.inf

    def test_nan_argument_decodes_to_nan(self):
        """A NaN argument reaches the worker as a real NaN float."""
        serializer = _serializer(float)
        decoded = serializer.decode_query(_through_the_strict_wire(serializer.encode_query({"payload": math.nan})))
        assert math.isnan(decoded["payload"])

    def test_infinity_argument_decodes_to_infinity(self):
        """An infinity argument reaches the worker as positive infinity."""
        serializer = _serializer(float)
        decoded = serializer.decode_query(_through_the_strict_wire(serializer.encode_query({"payload": math.inf})))
        assert decoded["payload"] == math.inf

    def test_nan_argument_inside_a_dict_decodes_to_nan(self):
        """A NaN nested inside a dict argument reaches the worker as a real NaN float."""
        serializer = _serializer(dict[str, float])
        decoded = serializer.decode_query(
            _through_the_strict_wire(serializer.encode_query({"payload": {"a": math.nan}}))
        )
        assert math.isnan(decoded["payload"]["a"])

    def test_nan_argument_inside_a_model_field_decodes_to_nan(self):
        """A NaN nested in a model argument field reaches the worker as a real NaN float."""
        serializer = _serializer(Inner)
        decoded = serializer.decode_query(
            _through_the_strict_wire(serializer.encode_query({"payload": Inner(name="x", score=math.nan)}))
        )
        assert math.isnan(decoded["payload"].score)


@pytest.mark.parametrize("value", [0.0, -0.0, 1e308, -1e308, 1.5])
class TestFiniteFloatsAreUntouched:
    def test_finite_float_result_is_encoded_as_a_plain_float(self, value):
        """A finite float result encodes to the same plain float without any marker."""
        serializer = _serializer(float)
        encoded = serializer.encode_result(value)
        assert isinstance(encoded, float)
        assert NON_FINITE_FLOAT_TAG not in json.dumps(encoded)

    def test_finite_float_result_survives_the_wire_unchanged(self, value):
        """A finite float result comes back with the exact same value and sign."""
        serializer = _serializer(float)
        decoded = serializer.decode_result(_through_the_strict_wire(serializer.encode_result(value)))
        assert _equal_including_non_finite_floats(decoded, value)

    def test_finite_float_argument_is_encoded_without_any_marker(self, value):
        """A finite float argument encodes without any marker key on the wire."""
        serializer = _serializer(float)
        assert NON_FINITE_FLOAT_TAG not in json.dumps(serializer.encode_query({"payload": value}))


class TestMarkerLookalikes:
    @pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
    def test_string_that_looks_like_a_token_stays_a_string(self, value):
        """A string spelled like a non-finite token stays a string instead of being revived as a float."""
        serializer = _serializer(str)
        decoded = serializer.decode_result(_through_the_strict_wire(serializer.encode_result(value)))
        assert decoded == value

    def test_token_string_inside_a_dict_stays_a_string(self):
        """A token-shaped string nested in a dict stays a string rather than becoming a float."""
        serializer = _serializer(dict[str, str])
        decoded = serializer.decode_result(_through_the_strict_wire(serializer.encode_result({"a": "nan"})))
        assert decoded == {"a": "nan"}

    def test_marker_shaped_dict_with_extra_keys_stays_a_dict(self):
        """A dict carrying the marker key alongside other keys decodes as a plain dict."""
        serializer = _serializer(dict)
        payload = {NON_FINITE_FLOAT_TAG: "nan", "extra": 1}
        assert serializer.decode_result(payload) == payload

    def test_marker_shaped_dict_with_unknown_token_stays_a_dict(self):
        """A dict carrying the marker key with an unknown token decodes as a plain dict."""
        serializer = _serializer(dict)
        payload = {NON_FINITE_FLOAT_TAG: "not-a-token"}
        assert serializer.decode_result(payload) == payload

    def test_marker_shaped_dict_with_extra_keys_stays_a_dict_as_an_argument(self):
        """A marker-shaped argument dict with extra keys reaches the worker as a plain dict."""
        serializer = _serializer(dict)
        payload = {NON_FINITE_FLOAT_TAG: "nan", "extra": 1}
        assert serializer.decode_query({"payload": payload})["payload"] == payload


@pytest.mark.parametrize("case", _BYTES_CASES, ids=[case[0] for case in _BYTES_CASES])
class TestBytesOnTheWire:
    def test_bytes_result_survives_the_wire(self, case):
        """Arbitrary bytes results come back byte for byte after base64 transport."""
        _, annotation, value = case
        serializer = _serializer(annotation)
        assert serializer.decode_result(_through_the_wire(serializer.encode_result(value))) == value

    def test_bytes_argument_survives_the_wire(self, case):
        """Arbitrary bytes arguments reach the worker byte for byte after base64 transport."""
        _, annotation, value = case
        serializer = _serializer(annotation)
        decoded = serializer.decode_query(_through_the_wire(serializer.encode_query({"payload": value})))
        assert decoded["payload"] == value

    def test_encoded_bytes_are_json_encodable(self, case):
        """Encoding bytes yields a json encodable payload rather than raw python bytes."""
        _, annotation, value = case
        serializer = _serializer(annotation)
        assert json.dumps(serializer.encode_result(value), allow_nan=False)


class TestBytesEncoding:
    def test_encoded_non_utf8_bytes_result_is_a_string(self):
        """A non-utf8 bytes result encodes to a base64 string on the wire."""
        serializer = _serializer(bytes)
        assert isinstance(serializer.encode_result(b"\x00\x80\xff"), str)

    def test_encoded_non_utf8_bytes_argument_is_a_string(self):
        """A non-utf8 bytes argument encodes to a base64 string on the wire."""
        serializer = _serializer(bytes)
        assert isinstance(serializer.encode_query({"payload": b"\x00\x80\xff"})["payload"], str)

    def test_utf8_bytes_nested_in_a_model_survive_the_wire(self):
        """Utf8-decodable bytes stored in a model field come back byte for byte on the revived model."""
        serializer = _serializer(Blob)
        decoded = serializer.decode_result(_through_the_wire(serializer.encode_result(Blob(data=b"payload"))))
        assert decoded.data == b"payload"

    def test_non_utf8_bytes_nested_in_a_model_are_refused_loudly(self):
        """A model carries its own serialization config, so non-utf8 bytes in a model field fail instead of corrupting."""
        serializer = _serializer(Blob)
        with pytest.raises(UnicodeDecodeError):
            serializer.encode_result(Blob(data=b"\x00\x80\xff"))


class TestRejectedPayloads:
    def test_result_containing_the_reserved_marker_key_is_rejected(self):
        """A result dict already using the reserved marker key fails loudly instead of being ambiguous."""
        serializer = _serializer(dict)
        with pytest.raises(ValueError, match=NON_FINITE_FLOAT_TAG):
            serializer.encode_result({NON_FINITE_FLOAT_TAG: "nan"})

    def test_result_using_the_reserved_marker_key_alongside_others_is_rejected(self):
        """The reserved key is refused even when it shares a dict with unrelated keys."""
        serializer = _serializer(dict)
        with pytest.raises(ValueError, match=NON_FINITE_FLOAT_TAG):
            serializer.encode_result({NON_FINITE_FLOAT_TAG: "nan", "extra": 1})

    def test_result_using_the_reserved_marker_key_with_an_unknown_token_is_rejected(self):
        """The reserved key is refused whatever value it carries, not only recognized tokens."""
        serializer = _serializer(dict)
        with pytest.raises(ValueError, match=NON_FINITE_FLOAT_TAG):
            serializer.encode_result({NON_FINITE_FLOAT_TAG: "not-a-token"})

    def test_nested_result_containing_the_reserved_marker_key_is_rejected(self):
        """A result carrying the reserved marker key deep inside is rejected too."""
        serializer = _serializer(dict)
        with pytest.raises(ValueError):
            serializer.encode_result({"outer": [{NON_FINITE_FLOAT_TAG: "inf"}]})

    def test_argument_containing_the_reserved_marker_key_is_rejected(self):
        """An argument dict already using the reserved marker key fails loudly instead of being ambiguous."""
        serializer = _serializer(dict)
        with pytest.raises(ValueError):
            serializer.encode_query({"payload": {NON_FINITE_FLOAT_TAG: "nan"}})

    def test_unserializable_result_is_rejected(self):
        """A result that is not json encodable fails rather than being silently coerced."""
        serializer = _serializer(int)
        with pytest.raises((TypeError, PydanticSerializationError)):
            json.dumps(serializer.encode_result(object()))

    def test_wrong_result_type_is_rejected_at_encode(self):
        """The worker that produced the bad result is where the failure lands, not the caller."""
        serializer = _serializer(int)
        with pytest.raises(PydanticSerializationError):
            serializer.encode_result("not-an-int")

    def test_wrong_result_type_is_rejected(self):
        """A result payload that does not match the annotation fails validation."""
        serializer = _serializer(int)
        with pytest.raises(ValidationError):
            serializer.decode_result("not-an-int")

    def test_extra_model_field_is_rejected(self):
        """Strict models refuse unknown fields arriving from the wire."""
        serializer = _serializer(Inner)
        with pytest.raises(ValidationError):
            serializer.decode_result({"name": "x", "score": 1.0, "extra": 1})

    def test_missing_model_field_is_rejected(self):
        """Strict models refuse payloads missing declared fields."""
        serializer = _serializer(Inner)
        with pytest.raises(ValidationError):
            serializer.decode_result({"name": "x"})

    def test_unknown_argument_is_rejected(self):
        """An argument the method does not declare is refused rather than ignored."""
        serializer = _serializer(int)
        with pytest.raises(ValidationError):
            serializer.decode_query({"payload": 1, "unknown": 2})

    def test_wrong_argument_type_is_rejected(self):
        """An argument that does not match the annotation fails validation."""
        serializer = _serializer(Inner)
        with pytest.raises(ValidationError):
            serializer.decode_query({"payload": "not-a-model"})
