import dataclasses
import datetime
import enum
import json
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


class Blob(StrictBaseModel):
    data: bytes


class Options(TypedDict):
    retries: int
    label: str


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
    def test_unserializable_result_is_rejected(self):
        """A result that is not json encodable fails rather than being silently coerced."""
        serializer = _serializer(int)
        with pytest.raises((TypeError, PydanticSerializationError)):
            json.dumps(serializer.encode_result(object()))

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
