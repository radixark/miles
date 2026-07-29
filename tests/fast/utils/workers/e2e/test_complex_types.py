import datetime
import uuid
from decimal import Decimal

import pytest
from pydantic import ValidationError

from tests.fast.utils.workers.e2e.e2e_worker import Colour, Item, Nested, Point


class TestComplexArgumentsAndResults:
    async def test_nested_model_roundtrips(self, handle):
        """A model containing models, dicts and sets survives the real wire both ways."""
        payload = Nested(
            item=Item(name="a", values=[1, 2]),
            lookup={"k": Item(name="b", values=[])},
            tags={"x", "y"},
        )
        assert await handle.demo_nested_model(payload=payload) == payload

    async def test_nested_model_result_is_revived_as_models(self, handle):
        """The nested result arrives as model instances, not plain dicts."""
        payload = Nested(item=Item(name="a", values=[1]), lookup={"k": Item(name="b", values=[2])}, tags=set())
        result = await handle.demo_nested_model(payload=payload)
        assert isinstance(result, Nested)
        assert isinstance(result.item, Item)
        assert isinstance(result.lookup["k"], Item)

    async def test_set_field_is_revived_as_a_set(self, handle):
        """A set field arrives as a set even though json carries a list."""
        payload = Nested(item=Item(name="a", values=[]), lookup={}, tags={"x", "y"})
        result = await handle.demo_nested_model(payload=payload)
        assert result.tags == {"x", "y"}
        assert isinstance(result.tags, set)

    async def test_enum_roundtrips_as_the_member(self, handle):
        """An enum argument and result stay enum members across the wire."""
        result = await handle.demo_enum(colour=Colour.BLUE)
        assert result is Colour.BLUE

    async def test_dataclass_roundtrips_as_the_dataclass(self, handle):
        """A dataclass argument is revived worker-side and the result comes back as a dataclass."""
        result = await handle.demo_dataclass(point=Point(x=1, y=2))
        assert isinstance(result, Point)
        assert (result.x, result.y) == (2, 1)

    async def test_aware_datetime_keeps_its_offset(self, handle):
        """A timezone-aware datetime keeps its instant and offset across the wire."""
        value = datetime.datetime(2026, 7, 27, 12, 30, tzinfo=datetime.timezone.utc)
        result = await handle.demo_datetime(when=value)
        assert isinstance(result, datetime.datetime)
        assert result == value

    async def test_naive_datetime_stays_naive(self, handle):
        """A naive datetime does not acquire a timezone on the way through."""
        value = datetime.datetime(2026, 7, 27, 12, 30)
        result = await handle.demo_datetime(when=value)
        assert result == value
        assert result.tzinfo is None

    async def test_uuid_roundtrips_as_a_uuid(self, handle):
        """A uuid argument and result stay uuid objects, not strings."""
        value = uuid.uuid4()
        result = await handle.demo_uuid(value=value)
        assert isinstance(result, uuid.UUID)
        assert result == value

    async def test_decimal_keeps_precision(self, handle):
        """A decimal keeps precision a float would lose."""
        value = Decimal("0.1234567890123456789")
        assert await handle.demo_decimal(value=value) == value

    async def test_tuple_roundtrips_as_a_tuple(self, handle):
        """A tuple result comes back as a tuple even though json carries a list."""
        result = await handle.demo_tuple(pair=(1, "a"))
        assert isinstance(result, tuple)
        assert result == (1, "a")

    @pytest.mark.parametrize("blob", [b"raw-bytes", b"", b"\x00\x80\xff", bytes(range(256))])
    async def test_bytes_roundtrip(self, handle, blob: bytes):
        """A bytes argument and result stay bytes across a text protocol, including non-utf8 payloads."""
        assert await handle.demo_bytes(blob=blob) == blob

    async def test_bytes_inside_a_list_roundtrips(self, handle):
        """Non-utf8 bytes nested in a container survive too."""
        blobs = [b"\x80", b"", b"\xff\xfe"]
        assert await handle.demo_bytes_list(blobs=blobs) == blobs

    async def test_list_of_models_roundtrips(self, handle):
        """A list of models arrives as a list of model instances."""
        items = [Item(name="a", values=[1]), Item(name="b", values=[2, 3])]
        result = await handle.demo_model_list(items=items)
        assert result == items
        assert all(isinstance(item, Item) for item in result)

    async def test_empty_containers_roundtrip(self, handle):
        """Empty containers survive rather than collapsing to null."""
        result = await handle.demo_model_list(items=[])
        assert result == []

    async def test_optional_present_and_absent(self, handle):
        """An optional argument roundtrips both with a value and with None."""
        assert await handle.demo_optional(value=7) == 7
        assert await handle.demo_optional(value=None) is None

    async def test_union_keeps_the_member_type(self, handle):
        """A union result keeps the member type it was produced with."""
        assert await handle.demo_union(value=5) == 5
        assert await handle.demo_union(value="five") == "five"

    async def test_union_of_types_sharing_a_wire_form_resolves_to_str(self, handle):
        """A datetime|str union shares one wire form, so the worker receives the str member."""
        when = datetime.datetime(2026, 7, 27, tzinfo=datetime.timezone.utc)
        assert await handle.report_union_argument_type(value=when) == "str"
        assert await handle.report_union_argument_type(value="plain") == "str"

    async def test_unicode_survives_the_wire(self, handle):
        """Non-ascii text roundtrips byte for byte."""
        text = "中文 🚀 \\ \" '"
        result = await handle.demo_model(item=Item(name=text, values=[]))
        assert result.name == text

    async def test_large_nested_payload_roundtrips(self, handle):
        """A large nested payload roundtrips without truncation."""
        payload = Nested(
            item=Item(name="big", values=list(range(5000))),
            lookup={f"k{i}": Item(name=f"n{i}", values=[i]) for i in range(200)},
            tags={f"t{i}" for i in range(200)},
        )
        assert await handle.demo_nested_model(payload=payload) == payload

    async def test_none_result_roundtrips(self, handle):
        """A method declared to return None reports success carrying None."""
        assert await handle.demo_none_result() is None


class TestWorkerSideTypes:
    async def test_worker_receives_model_instances(self, handle):
        """The worker is handed revived models, not the raw json dicts."""
        payload = Nested(item=Item(name="a", values=[1]), lookup={"k": Item(name="b", values=[])}, tags={"x"})
        assert await handle.report_nested_argument_types(payload=payload) == ["Nested", "Item", "Item", "set", "int"]

    async def test_worker_receives_a_dataclass_instance(self, handle):
        """A dataclass argument reaches the worker as the dataclass."""
        assert await handle.report_dataclass_argument_type(point=Point(x=1, y=2)) == "Point"

    async def test_worker_receives_the_enum_member(self, handle):
        """An enum argument reaches the worker as the member, comparable with is."""
        assert await handle.report_enum_argument_is_member(colour=Colour.BLUE) is True

    async def test_worker_receives_revived_scalars(self, handle):
        """Scalars json cannot express reach the worker as their declared python types."""
        types = await handle.report_scalar_argument_types(
            when=datetime.datetime(2026, 7, 27, tzinfo=datetime.timezone.utc),
            value=uuid.uuid4(),
            amount=Decimal("1.5"),
            blob=b"x",
            pair=(1, "a"),
        )
        assert types == ["datetime", "UUID", "Decimal", "bytes", "tuple"]


class TestComplexTypeValidation:
    async def test_wrong_model_type_is_rejected_before_sending(self, handle):
        """A payload of the wrong shape is rejected client-side, not by the worker."""
        with pytest.raises(ValidationError):
            await handle.demo_nested_model(payload={"item": {"name": "a"}})

    async def test_unknown_enum_member_is_rejected(self, handle):
        """A value outside the enum is refused rather than sent as a bare string."""
        with pytest.raises(ValidationError):
            await handle.demo_enum(colour="green")

    async def test_wrong_tuple_arity_is_rejected(self, handle):
        """A tuple of the wrong arity is refused client-side."""
        with pytest.raises(ValidationError):
            await handle.demo_tuple(pair=(1, "a", 2))

    async def test_extra_model_field_is_rejected(self, handle):
        """An unknown field inside a model argument is refused."""
        with pytest.raises(ValidationError):
            await handle.demo_model(item={"name": "a", "values": [], "extra": 1})

    async def test_worker_stays_usable_after_a_rejected_call(self, handle):
        """A client-side rejection does not consume anything server-side."""
        with pytest.raises(ValidationError):
            await handle.demo_enum(colour="green")
        assert await handle.demo_enum(colour=Colour.RED) is Colour.RED
