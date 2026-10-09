"""Independent boundary fixtures for the hard pilot's executable rules."""

from collections import Counter

from examples.clef.rl_pilot.hard import make_case, oracle, resolve


def check() -> None:
    rows = [{"entity": "x", "key": "amount", "value": value, "day": day, "signed": signed} for value, day, signed in [(10, 1, True), (20, 3, True), (30, 4, False), (40, 8, True)]]
    rows.append({"entity": "other", "key": "amount", "value": 99, "day": 5, "signed": True})
    assert resolve(rows, 5, "x") == {"amount": 20}
    assert resolve(rows, 2, "x") == {"amount": 10}
    p = {"tolerance": 5, "bond": 50, "priority": ["amount mismatch", "vendor verification", "incomplete delivery"], "routes": {"duplicate": "audit", "amount mismatch": "payments", "vendor verification": "risk", "incomplete delivery": "supply", "none": "none"}}
    v = {"qty1": 2, "price1": 100, "qty2": 3, "price2": 200, "credit": 20, "shipping": 10, "invoice": 795, "received1": 2, "received2": 3, "verified": True, "bank_changed": False, "bank_approved": False, "split_allowed": True, "bond": 50, "paid": False, "authority": 790}
    assert oracle("invoice", v, p) == {"action": "approve", "reason": "none", "owner": "none", "release": "yes"}
    assert oracle("invoice", {**v, "invoice": 796, "verified": False}, p)["reason"] == "amount mismatch"
    assert oracle("invoice", {**v, "paid": True, "verified": False}, p)["reason"] == "duplicate"
    assert oracle("invoice", {**v, "received1": 1}, p)["action"] == "approve"
    assert oracle("invoice", {**v, "received1": 1}, {**p, "variant": "heldout"})["action"] == "hold"
    v = {"request_day": 50, "delivered_day": 20, "damage_report": True, "photo_verified": True, "proof": True, "premium": True, "exception_signed": False, "excluded": False, "opened": True, "value": 100, "stock": 2, "requested": 2}
    p = {"damage_days": 30, "return_days": 30, "exception_days": 45, "return_threshold": 100}
    assert oracle("support", v, p) == {"action": "replace", "route": "quality", "return": "no", "expedite": "yes"}
    assert oracle("support", {**v, "proof": False}, p)["action"] == "request proof"
    assert oracle("support", v, {**p, "variant": "heldout"})["action"] == "deny"
    v = {"balance": 1000, "settled_credit": 200, "pending_hold": 190, "amount": 1000, "verified": True, "consent": True, "foreign": False, "approval_signed": True, "approval_limit": 2000, "beneficiary": "x", "approval_beneficiary": "x"}
    p = {"fee_bps": 1, "flat_fee": 9, "approval_threshold": 1010, "priority": ["reject", "verify identity", "request consent", "request approval"], "routes": {"execute": "payments", "reject": "billing", "verify identity": "identity", "request consent": "customer", "request approval": "compliance"}}
    assert oracle("agent", v, p)["fee"] == "10"
    assert oracle("agent", v, p)["action"] == "execute"
    assert oracle("agent", {**v, "pending_hold": 191}, p)["action"] == "reject"
    v = {"operation": "refund", "region": "west", "amount": 50, "authorized": True, "has_id": False}
    capable = {"name": "t1", "operation": "refund", "regions": ["west"], "max_amount": 50, "requires_auth": True, "requires_id": False, "mutation": True, "cost": 2}
    p = {"tools": [capable, {**capable, "name": "t2", "cost": 1, "requires_id": True}], "approval_limit": 50}
    assert oracle("tool", v, p) == {"next": "t1", "mutates": "yes", "approval": "no"}
    assert oracle("tool", {**v, "has_id": True}, p)["next"] == "t2"
    counts = Counter()
    for split, n in [("train", 2048), ("validation", 256)]:
        groups = set()
        for index in range(n):
            case = make_case(index, split, 261010)
            assert case["scenario_group"] not in groups
            groups.add(case["scenario_group"])
            for key, answer in case["answers"].items():
                options = case["questions"][key]["criteria"]
                assert len(options) >= 2 and len(set(options.values())) == len(options) and answer in options
                counts[(split, case["family"], key, options[answer])] += 1
    print("PASS: version eligibility, arithmetic boundaries, exception/precedence, tool capability/cost, all2304 schemas and unique scenarios")
    print("LABEL_COUNTS", dict(counts))


if __name__ == "__main__":
    check()
