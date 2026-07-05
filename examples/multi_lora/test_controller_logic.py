"""Fast tests for MultiLoRAControllerLogic (no Ray, no HTTP, no SGLang)."""

from examples.multi_lora.controller import MultiLoRAControllerLogic, make_rid, parse_adapter


def test_rid_roundtrip_preserves_names_with_underscores():
    for name in ["a", "adapter_a", "weird__name", "x_y_z"]:
        assert parse_adapter(make_rid(name)) == name


def test_forward_active_then_response_kept():
    c = MultiLoRAControllerLogic()
    c.register("A", 0)
    rid = make_rid("A")
    assert c.on_forward(rid) is True
    assert c.on_response(rid) is False  # A still active -> keep


def test_forward_blocked_for_unknown_adapter():
    c = MultiLoRAControllerLogic()
    assert c.on_forward(make_rid("A")) is False  # never registered -> block


def test_deregister_mid_flight_dummies_response():
    c = MultiLoRAControllerLogic()
    c.register("A", 0)
    rid = make_rid("A")
    assert c.on_forward(rid) is True
    c.deregister("A")  # swapped out mid-flight
    assert c.on_response(rid) is True  # A gone -> dummy


def test_deregister_then_new_request_blocked():
    c = MultiLoRAControllerLogic()
    c.register("A", 0)
    c.deregister("A")
    assert c.on_forward(make_rid("A")) is False


def test_swap_a_to_b_independent():
    c = MultiLoRAControllerLogic()
    c.register("A", 0)
    rid_a = make_rid("A")
    assert c.on_forward(rid_a) is True
    c.deregister("A")
    c.register("B", 0)  # reuse slot 0
    rid_b = make_rid("B")
    assert c.on_forward(rid_b) is True  # B active -> forward
    assert c.on_response(rid_a) is True  # straggler A -> dummy
    assert c.on_response(rid_b) is False  # B -> keep


def test_register_stores_config_and_active_adapters():
    c = MultiLoRAControllerLogic()
    c.register("A", 0, config={"rm_type": "x"})
    assert c.active_adapters() == {"A": {"slot": 0, "config": {"rm_type": "x"}}}
    c.deregister("A")
    assert c.active_adapters() == {}
