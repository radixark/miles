"""Fast tests for MultiLoRAControllerLogic (no Ray, no HTTP, no SGLang, no torch)."""

from miles.utils.adapter_config import AdapterState

from miles.utils.multi_lora import MultiLoRAControllerLogic, make_rid, parse_adapter


def test_rid_roundtrip_preserves_names_with_underscores():
    for name in ["a", "adapter_a", "weird__name", "x_y_z"]:
        assert parse_adapter(make_rid(name)) == name


def test_register_assigns_slot_and_active_adapters_view():
    c = MultiLoRAControllerLogic(max_adapters=4)
    result = c.register_adapter("A", config={"rm_type": "x"})
    assert result == {"name": "A", "slot": 0}
    assert c.active() == {"A": 0}
    view = c.active_adapters()["A"]
    assert view.name == "A"
    assert view.slot == 0
    assert view.config == {"rm_type": "x"}
    assert view.state == AdapterState.RUNNING


def test_forward_active_then_response_kept():
    c = MultiLoRAControllerLogic(max_adapters=4)
    c.register_adapter("A", None)
    rid = make_rid("A")
    assert c.on_forward(rid) is True
    assert c.on_response(rid) is False  # A still active -> keep


def test_forward_blocked_for_unknown_adapter():
    c = MultiLoRAControllerLogic(max_adapters=4)
    assert c.on_forward(make_rid("A")) is False  # never registered -> block


def test_deregister_mid_flight_dummies_response():
    c = MultiLoRAControllerLogic(max_adapters=4)
    c.register_adapter("A", None)
    rid = make_rid("A")
    assert c.on_forward(rid) is True
    c.deregister_adapter("A")  # removed mid-flight
    assert c.on_response(rid) is True  # A gone -> dummy


def test_deregister_then_new_request_blocked():
    c = MultiLoRAControllerLogic(max_adapters=4)
    c.register_adapter("A", None)
    c.deregister_adapter("A")
    assert c.on_forward(make_rid("A")) is False


def test_deregister_holds_slot_until_free_slot():
    c = MultiLoRAControllerLogic(max_adapters=2)
    c.register_adapter("A", None)  # slot 0
    c.register_adapter("B", None)  # slot 1
    c.deregister_adapter("A")  # slot 0 held, not freed
    assert not c.free_slots  # no free slots (0 held, 1 in use)
    c.free_slot("A")  # trainer cleanup -> slot 0 freed
    assert c.register_adapter("C", None) == {"name": "C", "slot": 0}  # reuses freed slot
    assert c.active() == {"B": 1, "C": 0}


def test_swap_a_to_b_independent():
    c = MultiLoRAControllerLogic(max_adapters=4)
    c.register_adapter("A", None)
    rid_a = make_rid("A")
    assert c.on_forward(rid_a) is True
    c.deregister_adapter("A")
    c.register_adapter("B", None)  # reuses slot 0
    rid_b = make_rid("B")
    assert c.on_forward(rid_b) is True  # B active -> forward
    assert c.on_response(rid_a) is True  # straggler A -> dummy
    assert c.on_response(rid_b) is False  # B -> keep
