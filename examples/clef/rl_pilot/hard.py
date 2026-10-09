"""Versioned evidence and compositional policies with executable exact labels."""

import asyncio
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from tap import Tap

from examples.clef.rl_pilot.generate import field, generate_one


class Args(Tap):
    output: Path
    key_file: Path = Path("/home/ubuntu/openai.key")
    count: int = 2048
    validation_count: int = 256
    concurrency: int = 16
    seed: int = 261010
    canonical_only: bool = False
    audit_effort: str = "medium"
    audit_tokens: int = 4800


def resolve(rows: list[dict], as_of: int, entity: str) -> dict[str, Any]:
    eligible = [r for r in rows if r["entity"] == entity and r["day"] <= as_of and r["signed"]]
    result = {}
    for key in {r["key"] for r in eligible}:
        choices = [r for r in eligible if r["key"] == key]
        result[key] = max(choices, key=lambda r: r["day"])["value"]
    return result


def oracle(family: str, values: dict, policy: dict) -> dict[str, str]:
    """Pure business evaluator; no API response is used to construct labels."""
    values = dict(values)
    if policy.get("variant") == "heldout":
        if family == "invoice":
            values["split_allowed"] = values["split_allowed"] and values["bank_approved"]
        elif family == "support":
            values["photo_verified"] = values["photo_verified"] and values["exception_signed"]
        elif family == "security":
            values["manager_signed"] = values["manager_signed"] and values["exception_scope"] == "device"
        elif family == "agent":
            values["approval_signed"] = values["approval_signed"] and values["consent"]
        elif family == "tool":
            values["authorized"] = values["authorized"] and values["has_id"]
        elif family == "retrieval":
            values["opened"] = values["opened"] or values["tier"] == "basic"
    if family == "invoice":
        base = sum(values[f"qty{i}"] * values[f"price{i}"] for i in (1, 2))
        expected = base - values["credit"] + values["shipping"]
        difference = abs(values["invoice"] - expected)
        missing = values["received1"] < values["qty1"] or values["received2"] < values["qty2"]
        holds = []
        if difference > policy["tolerance"]:
            holds.append("amount mismatch")
        if missing and not (values["split_allowed"] and values["bond"] >= policy["bond"]):
            holds.append("incomplete delivery")
        if not values["verified"] or values["bank_changed"] and not values["bank_approved"]:
            holds.append("vendor verification")
        reason = "duplicate" if values["paid"] else next((x for x in policy["priority"] if x in holds), "none")
        action = "reject" if reason == "duplicate" else "hold" if holds else "approve"
        return {"action": action, "reason": reason, "owner": policy["routes"][reason], "release": "yes" if action == "approve" and expected <= values["authority"] else "no"}
    if family == "support":
        elapsed = values["request_day"] - values["delivered_day"]
        damage = values["damage_report"] and values["photo_verified"]
        timely_damage = damage and elapsed <= policy["damage_days"]
        timely_return = not values["opened"] and elapsed <= policy["return_days"]
        exception = values["premium"] and values["exception_signed"] and elapsed <= policy["exception_days"]
        action = "request proof" if not values["proof"] else "deny" if values["excluded"] and not exception else "replace" if timely_damage else "refund" if timely_return or exception else "deny"
        route = "quality" if timely_damage and values["proof"] else "specialist" if exception and values["excluded"] else "billing"
        return {"action": action, "route": route, "return": "yes" if action in {"replace", "refund"} and values["value"] > policy["return_threshold"] else "no", "expedite": "yes" if action == "replace" and values["premium"] and values["stock"] >= values["requested"] else "no"}
    if family == "security":
        score = policy["failure_weight"] * values["failures"] + policy["geo_weight"] * int(values["geo_changed"]) - policy["trust_discount"] * int(values["trusted"])
        exception = values["exception_signed"] and values["exception_expiry"] >= values["as_of"] and values["exception_scope"] == "device"
        hard_block = not values["mfa"] or values["revoked"] or score >= policy["block"]
        review = score >= policy["review"] or not values["device_approved"] and not exception
        action = "block" if hard_block else "review" if review else "allow"
        return {
            "action": action,
            "team": {"block": "incident response", "review": "identity", "allow": "none"}[action],
            "elevation": "yes" if action == "allow" and values["elevation_requested"] and values["manager_signed"] and values["role"] in policy["roles"] else "no",
            "notify": "yes" if action != "allow" or values["geo_changed"] and not values["trusted"] else "no",
        }
    if family == "agent":
        available = values["balance"] + values["settled_credit"] - values["pending_hold"]
        fee = (values["amount"] * policy["fee_bps"] + 9999) // 10000 + policy["flat_fee"]
        total = values["amount"] + fee
        approved = values["approval_signed"] and values["approval_limit"] >= total and values["approval_beneficiary"] == values["beneficiary"]
        required = total > policy["approval_threshold"] or values["foreign"]
        conditions = {"verify identity": not values["verified"], "request consent": not values["consent"], "reject": total > available, "request approval": required and not approved}
        action = next((a for a in policy["priority"] if conditions[a]), "execute")
        return {"action": action, "owner": policy["routes"][action], "debit": "yes" if action == "execute" else "no", "fee": str(fee)}
    if family == "tool":
        eligible = []
        for tool in policy["tools"]:
            if tool["operation"] == values["operation"] and values["region"] in tool["regions"] and tool["max_amount"] >= values["amount"]:
                if (not tool["requires_auth"] or values["authorized"]) and (not tool["requires_id"] or values["has_id"]):
                    eligible.append(tool)
        chosen = min(eligible, key=lambda t: t["cost"]) if eligible else None
        action = chosen["name"] if chosen else "ask for missing authorization or inputs" if not values["authorized"] or not values["has_id"] else "escalate: no capable tool"
        return {"next": action, "mutates": "yes" if chosen and chosen["mutation"] else "no", "approval": "yes" if chosen and chosen["mutation"] and values["amount"] > policy["approval_limit"] else "no"}
    if family == "retrieval":
        applicable = [doc for doc in policy["docs"] if doc["region"] == values["region"] and doc["tier"] == values["tier"] and doc["effective"] <= values["as_of"] and doc["signed"]]
        chosen = max(applicable, key=lambda d: d["effective"])
        eligible = values["elapsed"] <= chosen["window"] and (not values["opened"] or chosen["opened_allowed"])
        return {"document": chosen["id"], "window": str(chosen["window"]), "eligible": "yes" if eligible else "no"}
    raise ValueError(f"unknown family: {family}")


def make_case(index: int, split: str, seed: int) -> dict[str, Any]:
    rng = random.Random(f"{seed}:{split}:{index}")
    family = ["invoice", "support", "security", "agent", "tool", "retrieval", "invoice", "support"][index % 8]
    identifier = f"hard-{split}-{index:05d}"
    entity = f"CASE-{rng.randrange(10**8, 10**9)}"
    as_of = 30
    values: dict[str, Any] = {}
    policy: dict[str, Any] = {}
    options: dict[str, list[str]] = {}
    # Validation uses held-out compositions, not only new random document IDs.
    variant = "heldout" if split == "validation" else "standard"
    if family == "invoice":
        values = {
            "qty1": rng.randint(3, 20),
            "qty2": rng.randint(4, 15),
            "price1": rng.randint(80, 900),
            "price2": rng.randint(50, 700),
            "credit": rng.randint(0, 300),
            "shipping": rng.randint(0, 100),
            "verified": rng.random() < 0.85,
            "bank_changed": rng.random() < 0.4,
            "bank_approved": rng.random() < 0.7,
            "paid": rng.random() < 0.12,
            "split_allowed": rng.random() < 0.5,
            "bond": rng.randint(0, 100),
            "authority": rng.randint(2000, 18000),
        }
        values["received1"] = values["qty1"] - rng.choice([0, 0, 1, 2])
        values["received2"] = values["qty2"] - rng.choice([0, 0, 1, 2])
        expected = sum(values[f"qty{i}"] * values[f"price{i}"] for i in (1, 2)) - values["credit"] + values["shipping"]
        policy = {"tolerance": rng.randint(1, 15), "bond": rng.randint(30, 70), "priority": ["amount mismatch", "vendor verification", "incomplete delivery"]}
        rng.shuffle(policy["priority"])
        reasons = ["duplicate", "amount mismatch", "vendor verification", "incomplete delivery", "none"]
        teams = ["payments", "vendor risk", "procurement", "audit", "none"]
        rng.shuffle(teams)
        policy["routes"] = dict(zip(reasons, teams, strict=True))
        values["invoice"] = expected + rng.choice([0, policy["tolerance"], policy["tolerance"] + 1, -policy["tolerance"] - 1])
        options = {"action": ["approve", "hold", "reject"], "reason": reasons, "owner": teams, "release": ["yes", "no"]}
        prose = (
            f"All money is integer cents. Expected invoice = qty1*price1 + qty2*price2 - credit + shipping. "
            f"A mismatch means absolute difference exceeds {policy['tolerance']} cents (equality is permitted). "
            f"Already paid means reject with reason duplicate. Otherwise hold for vendor verification if unverified OR bank changed without bank approval; "
            f"hold for incomplete delivery if either received quantity is below ordered quantity UNLESS split_allowed and bond >= {policy['bond']}; "
            f"hold for amount mismatch. If multiple hold reasons apply, choose the first in this priority list: {policy['priority']}. "
            f"Otherwise approve with reason none. Owner mapping: {policy['routes']}. Release payment only when approved AND expected amount <= authority."
        )
        if variant == "heldout":
            prose += " For this validation policy, split_allowed is the resolved split permission AND resolved bank approval."
    elif family == "support":
        values = {
            "request_day": rng.randint(40, 80),
            "delivered_day": rng.randint(10, 39),
            "proof": rng.random() < 0.85,
            "opened": rng.choice([True, False]),
            "damage_report": rng.choice([True, False]),
            "photo_verified": rng.random() < 0.65,
            "premium": rng.choice([True, False]),
            "exception_signed": rng.random() < 0.65,
            "excluded": rng.choice([True, False]),
            "value": rng.randint(500, 20000),
            "stock": rng.randint(0, 8),
            "requested": rng.randint(1, 6),
        }
        policy = {"damage_days": rng.randint(20, 55), "return_days": rng.randint(15, 40), "exception_days": rng.randint(45, 70), "return_threshold": rng.randint(3000, 10000)}
        options = {"action": ["request proof", "deny", "replace", "refund"], "route": ["quality", "specialist", "billing"], "return": ["yes", "no"], "expedite": ["yes", "no"]}
        prose = (
            f"Elapsed days = request_day - delivered_day, not today's cutoff minus delivered_day. Damage is verified only if both damage_report and photo_verified. "
            f"A premium exception requires premium AND exception_signed AND elapsed <= {policy['exception_days']}. "
            f"In order: missing proof -> request proof; excluded item without premium exception -> deny; "
            f"verified damage with elapsed <= {policy['damage_days']} -> replace; unopened with elapsed <= {policy['return_days']} OR premium exception -> refund; otherwise deny. "
            f"Route to quality if proof exists and verified damage is timely, else specialist if a premium exception applies to an excluded item, else billing. "
            f"Request return shipment only for refund/replace AND value > {policy['return_threshold']}. Expedite only for replace AND premium AND stock >= requested."
        )
        if variant == "heldout":
            prose += " This policy additionally requires exception_signed for photo verification to count."
    elif family == "security":
        values = {
            "failures": rng.randint(0, 12),
            "geo_changed": rng.choice([True, False]),
            "trusted": rng.choice([True, False]),
            "mfa": rng.random() < 0.85,
            "revoked": rng.random() < 0.15,
            "device_approved": rng.choice([True, False]),
            "exception_signed": rng.choice([True, False]),
            "exception_expiry": rng.randint(25, 35),
            "as_of": as_of,
            "exception_scope": rng.choice(["device", "location"]),
            "elevation_requested": rng.choice([True, False]),
            "manager_signed": rng.choice([True, False]),
            "role": rng.choice(["analyst", "operator", "administrator"]),
        }
        policy = {"failure_weight": rng.randint(2, 5), "geo_weight": rng.randint(4, 12), "trust_discount": rng.randint(3, 10), "block": rng.randint(30, 45), "review": rng.randint(12, 22), "roles": rng.sample(["analyst", "operator", "administrator"], 2)}
        options = {"action": ["allow", "review", "block"], "team": ["incident response", "identity", "none"], "elevation": ["yes", "no"], "notify": ["yes", "no"]}
        prose = (
            f"Risk = failures*{policy['failure_weight']} + ({policy['geo_weight']} if geo_changed else 0) - ({policy['trust_discount']} if trusted else 0). "
            f"A device exception requires exception_signed, expiry >= as_of, and exception_scope exactly device. It never waives a hard block or risk threshold. "
            f"Block for missing MFA, revoked account, or risk >= {policy['block']}. Otherwise review for risk >= {policy['review']} or unapproved device without valid device exception. Otherwise allow. "
            f"Teams: block=incident response, review=identity, allow=none. Grant elevation only if allow AND elevation_requested AND manager_signed AND role in {policy['roles']}. "
            "Notify if action is not allow OR (geo_changed AND not trusted)."
        )
        if variant == "heldout":
            prose += " This policy additionally requires exception_scope=device before a manager signature counts for elevation."
    elif family == "agent":
        values = {
            "balance": rng.randint(3000, 30000),
            "settled_credit": rng.randint(0, 4000),
            "pending_hold": rng.randint(0, 5000),
            "amount": rng.randint(1000, 35000),
            "verified": rng.random() < 0.85,
            "consent": rng.random() < 0.85,
            "approval_signed": rng.random() < 0.7,
            "approval_limit": rng.randint(5000, 40000),
            "beneficiary": "primary",
            "approval_beneficiary": rng.choice(["primary", "previous"]),
            "foreign": rng.choice([True, False]),
        }
        policy = {"fee_bps": rng.choice([15, 25, 40, 75, 100]), "flat_fee": rng.randint(20, 120), "approval_threshold": rng.randint(8000, 22000), "priority": ["verify identity", "request consent", "reject", "request approval"]}
        rng.shuffle(policy["priority"])
        actions = policy["priority"] + ["execute"]
        teams = ["identity", "customer", "billing", "compliance", "payments"]
        rng.shuffle(teams)
        policy["routes"] = dict(zip(actions, teams, strict=True))
        fee = (values["amount"] * policy["fee_bps"] + 9999) // 10000 + policy["flat_fee"]
        options = {"action": actions, "owner": teams, "debit": ["yes", "no"], "fee": [str(x) for x in [fee, fee + 1, fee - 1, fee + policy["flat_fee"], policy["flat_fee"]]]}
        options["fee"] = list(dict.fromkeys(options["fee"]))
        prose = (
            f"All amounts are integer cents. Available funds = balance + settled_credit - pending_hold. "
            f"Fee = ceiling(amount*{policy['fee_bps']}/10000) + {policy['flat_fee']}; total debit = amount + fee. "
            f"Approval required if total debit > {policy['approval_threshold']} OR foreign. Approval valid only if signed, approval_limit >= total debit, and approval_beneficiary equals beneficiary. "
            f"Find all applicable actions: unverified -> verify identity; no consent -> request consent; total debit > available -> reject; required but invalid approval -> request approval. "
            f"Choose FIRST applicable action in priority {policy['priority']}; if none, execute. Owner mapping {policy['routes']}. Debit only for execute."
        )
        if variant == "heldout":
            prose += " This policy additionally requires consent before approval_signed counts."
    elif family == "tool":
        values = {"operation": rng.choice(["refund", "lookup", "cancel"]), "region": rng.choice(["east", "west", "central"]), "amount": rng.randint(100, 1200), "authorized": rng.choice([True, False]), "has_id": rng.choice([True, False])}
        names = [f"tool_{rng.randrange(1000, 9999)}_{i}" for i in range(8)]
        tools = [
            {"name": n, "operation": rng.choice(["refund", "lookup", "cancel"]), "regions": rng.sample(["east", "west", "central"], rng.randint(1, 3)), "max_amount": rng.randint(200, 1400), "requires_auth": rng.choice([True, False]), "requires_id": rng.choice([True, False]), "cost": 10 * i + rng.randint(1, 9)}
            for i, n in enumerate(names)
        ]
        for tool in tools:
            tool["mutation"] = tool["operation"] != "lookup"
        rng.shuffle(tools)
        policy = {"tools": tools, "approval_limit": rng.randint(400, 900)}
        options = {"next": names + ["ask for missing authorization or inputs", "escalate: no capable tool"], "mutates": ["yes", "no"], "approval": ["yes", "no"]}
        prose = (
            "A capable tool must match requested operation, contain request region in its supported regions, and have max_amount >= amount. "
            "It is eligible now only if each required auth/ID input is present. Choose eligible tool with LOWEST cost; costs are unique. "
            "If none eligible, ask for missing authorization or inputs when authorized=false OR has_id=false; otherwise escalate: no capable tool. "
            f"mutates=yes only when the chosen tool is a mutation. approval=yes only when chosen tool is a mutation AND amount > {policy['approval_limit']}. "
            "Tool catalog is authoritative: " + json.dumps(tools, sort_keys=True)
        )
        if variant == "heldout":
            prose += " This policy additionally treats authorization as valid only when has_id=true."
    else:
        values = {"region": rng.choice(["east", "west", "central"]), "tier": rng.choice(["basic", "premium"]), "as_of": as_of, "elapsed": rng.randint(10, 60), "opened": rng.choice([True, False])}
        docs = [{"id": f"D{i}", "region": rng.choice(["east", "west", "central"]), "tier": rng.choice(["basic", "premium"]), "effective": i * 4, "signed": rng.choice([True, False]), "window": rng.randint(15, 55), "opened_allowed": rng.choice([True, False])} for i in range(9)]
        docs[0].update(region=values["region"], tier=values["tier"], signed=True)
        docs[4].update(region=values["region"], tier=values["tier"], signed=True)
        rng.shuffle(docs)
        policy = {"docs": docs}
        options = {"document": [d["id"] for d in docs], "window": [str(x) for x in sorted({d["window"] for d in docs})], "eligible": ["yes", "no"]}
        prose = (
            "Policy documents must match BOTH resolved region and tier, be signed, and be effective no later than as_of. "
            "Among eligible documents use the latest effective day, not the largest window. Return eligible only if elapsed <= chosen window AND "
            "(item is unopened OR chosen policy allows opened items). Policy catalog: " + json.dumps(docs, sort_keys=True)
        )
        if variant == "heldout":
            prose += " This policy treats all basic-tier items as opened regardless of reported opened status."
    # Deliberately exercise rare positive/negative branches, instead of letting
    # "no" dominate auxiliary fields as it did in the easy pilot.
    mode = (index // 8) % 4
    if family == "invoice":
        if mode == 0:
            values.update(paid=False, verified=True, bank_changed=False, received1=values["qty1"], received2=values["qty2"], invoice=expected, authority=expected + 1)
        elif mode == 1:
            values["paid"] = True
        elif mode == 2:
            values.update(paid=False, invoice=expected + policy["tolerance"] + 1)
        else:
            values.update(paid=False, verified=False)
    elif family == "support":
        if mode == 0:
            values.update(proof=True, excluded=False, premium=True, damage_report=True, photo_verified=True, exception_signed=True, request_day=values["delivered_day"] + policy["damage_days"], stock=values["requested"], value=policy["return_threshold"] + 1)
        elif mode == 1:
            values.update(proof=True, excluded=False, premium=False, damage_report=False, opened=False, request_day=values["delivered_day"] + policy["return_days"])
        elif mode == 2:
            values.update(proof=True, excluded=True, exception_signed=False)
        else:
            values["proof"] = False
    elif family == "security":
        if mode in (0, 1):
            values.update(failures=0, geo_changed=False, mfa=True, revoked=False, device_approved=True, elevation_requested=(mode == 0), manager_signed=True, exception_scope="device", role=policy["roles"][0])
        elif mode == 2:
            values.update(failures=0, geo_changed=False, mfa=True, revoked=False, device_approved=False, exception_signed=False)
        else:
            values["revoked"] = True
    elif family == "agent":
        if mode == 0:
            values.update(verified=True, consent=True, balance=100000, pending_hold=0, foreign=False, approval_signed=True, approval_limit=100000, approval_beneficiary=values["beneficiary"])
        elif mode == 1:
            values.update(verified=True, consent=True, balance=0, settled_credit=0, pending_hold=0, approval_signed=True, approval_limit=100000, approval_beneficiary=values["beneficiary"])
        elif mode == 2:
            values.update(verified=True, consent=True, balance=100000, foreign=True, approval_signed=False)
        else:
            values.update(verified=False, consent=False, balance=100000, approval_signed=True, approval_limit=100000, approval_beneficiary=values["beneficiary"])
        fee = (values["amount"] * policy["fee_bps"] + 9999) // 10000 + policy["flat_fee"]
        options["fee"] = list(dict.fromkeys(str(x) for x in [fee, fee + 1, fee - 1, fee + policy["flat_fee"], policy["flat_fee"]]))
    elif family == "tool":
        if mode in (0, 1):
            values.update(authorized=True, has_id=True)
            tools[0].update(operation=values["operation"], regions=[values["region"]], max_amount=values["amount"] + 1, mutation=values["operation"] != "lookup")
        elif mode == 2:
            values.update(authorized=True, has_id=True)
            for tool in tools:
                tool["max_amount"] = values["amount"] - 1
        else:
            values.update(authorized=False, has_id=False)
            for tool in tools:
                tool.update(requires_auth=True, requires_id=True)
        # The catalog is embedded in prose: rebuild it after branch balancing.
        prose = prose[: prose.index("Tool catalog is authoritative:")] + "Tool catalog is authoritative: " + json.dumps(tools, sort_keys=True)
        if variant == "heldout":
            prose += " This policy additionally treats authorization as valid only when has_id=true."
    else:
        eligible_docs = [doc for doc in docs if doc["region"] == values["region"] and doc["tier"] == values["tier"] and doc["effective"] <= as_of and doc["signed"]]
        chosen = max(eligible_docs, key=lambda d: d["effective"])
        values.update(elapsed=chosen["window"] if mode < 2 else chosen["window"] + 1, opened=False)
    policy["variant"] = variant
    # Evidence ledger contains signed revisions, unsigned proposals, future
    # revisions, and another entity. No resolved values are exposed separately.
    rows = []
    for key, value in values.items():
        alternative = not value if isinstance(value, bool) else value + rng.randint(1, 12) if isinstance(value, int) else "previous"
        latest_day = rng.randint(10, 22)
        rows.extend(
            [
                {"entity": entity, "key": key, "value": alternative, "day": latest_day - 3, "signed": True},
                {"entity": entity, "key": key, "value": value, "day": latest_day, "signed": True},
                {"entity": entity, "key": key, "value": alternative, "day": latest_day + 2, "signed": False},
                {"entity": entity, "key": key, "value": alternative, "day": 31 + rng.randint(1, 8), "signed": True},
            ]
        )
    for key in rng.sample(list(values), min(4, len(values))):
        rows.append({"entity": entity + "-OLD", "key": key, "value": values[key], "day": 29, "signed": True})
    rng.shuffle(rows)
    resolved = resolve(rows, as_of, entity)
    # Held-out transformations are explicit policy operations, applied after
    # resolving the ledger. Keep the raw ledger semantically distinct.
    if resolved != values:
        raise AssertionError("ledger does not reproduce canonical values")
    answers_semantic = oracle(family, resolved, policy)
    questions, answers = {}, {}
    for name, descriptions in options.items():
        rng.shuffle(descriptions)
        questions[name] = field(f"Determine {name} for {entity} using only the authoritative policy and eligible records.", descriptions)
        answers[name] = next(k for k, v in questions[name]["criteria"].items() if v == answers_semantic[name])
    facts = [f"Requested entity {entity}; evidence cutoff is day {as_of}. Every day is a relative integer; no outside date arithmetic is needed."]
    # Compact bundles reduce irrelevant token repetition without omitting any
    # signed/unsigned/future conflict or entity identifier.
    for offset in range(0, len(rows), 8):
        facts.append("Evidence ledger entries: " + json.dumps(rows[offset : offset + 8], sort_keys=True))
    global_rule = "For each key, use the most recent SIGNED entry for the requested entity with day <= evidence cutoff. Ignore unsigned, future, and other-entity entries. Resolve keys independently, THEN apply the business policy. Booleans are literal true/false. "
    payload = {"family": family, "rows": sorted(rows, key=lambda r: (r["key"], r["entity"], r["day"])), "policy": policy, "variant": variant}
    semantic_payload = json.dumps(payload, sort_keys=True).replace(entity, "CASE")
    return {
        "id": identifier,
        "family": family,
        "facts": facts,
        "policy": global_rule + prose,
        "questions": questions,
        "answers": answers,
        "split": split,
        "scenario_seed": f"{seed}:{split}:{index}",
        "scenario_group": hashlib.sha256(semantic_payload.encode()).hexdigest(),
        "policy_variant": variant,
        "oracle_inputs": {"rows": rows, "as_of": as_of, "entity": entity, "policy": policy},
    }


def canonical_example(case: dict) -> dict:
    return {
        "record": {"id": case["id"], "state": "Authoritative policy:\n" + case["policy"] + "\n\n" + "\n".join(case["facts"]), "questions": case["questions"]},
        "targets": {k: {o: float(o == answer) for o in case["questions"][k]["criteria"]} for k, answer in case["answers"].items()},
        "source": "clef_rl_hard_" + case["family"],
        "metadata": {"split": case["split"], "ground_truth": case},
    }


def validate_and_finalize(args: Args) -> None:
    report = {}
    groups = {}
    for split, count in (("train", args.count), ("validation", args.validation_count)):
        rows = []
        for index in range(count):
            case = make_case(index, split, args.seed)
            path = args.output / "accepted" / f"{case['id']}.json"
            row = json.loads(path.read_text())
            assert row["metadata"]["ground_truth"] == case
            assert row["record"]["questions"] == case["questions"]
            assert case["policy"] in row["record"]["state"]
            assert all(fact in row["record"]["state"] for fact in case["facts"])
            inputs = case["oracle_inputs"]
            semantic = oracle(case["family"], resolve(inputs["rows"], inputs["as_of"], inputs["entity"]), inputs["policy"])
            assert {k: case["questions"][k]["criteria"][v] for k, v in case["answers"].items()} == semantic
            for key, answer in case["answers"].items():
                assert row["targets"][key] == {o: float(o == answer) for o in case["questions"][key]["criteria"]}
            if not args.canonical_only:
                audit = row["metadata"]["audit"]
                assert audit["answers"] == semantic and audit["unambiguous"] and not audit["unsupported_claims"]
            row["source"] = "clef_rl_hard_" + case["family"]
            rows.append(row)
        assert len({r["record"]["id"] for r in rows}) == count
        assert len({r["record"]["state"] for r in rows}) == count
        groups[split] = {r["metadata"]["ground_truth"]["scenario_group"] for r in rows}
        assert len(groups[split]) == count
        random.Random(f"{args.seed}:{split}").shuffle(rows)
        destination = args.output / f"{split}.jsonl"
        destination.write_text("".join(json.dumps(r) + "\n" for r in rows))
        report[split] = {"records": count, "fields": sum(len(r["targets"]) for r in rows), "families": dict(Counter(r["source"] for r in rows)), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}
    assert not groups["train"] & groups["validation"]
    manifest = {
        "seed": args.seed,
        "train": args.count,
        "validation": args.validation_count,
        "model": "gpt-6-luna" if not args.canonical_only else "none",
        "sha256": {s: report[s]["sha256"] for s in report},
        "report": report,
        "checks": ["executable version resolver", "executable business oracle", "deterministic regeneration", "evidence preservation", "unique scenarios", "disjoint scenarios", "held-out policy compositions"],
        "limitations": ["Same-model blind review, not human verification.", "Shared task families and resolver between splits; held-out additional clause compositions are limited OOD coverage.", "No real tool execution or organic cases."],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.output / "validation-report.json").write_text(json.dumps(report, indent=2))
    print("COMPLETE", json.dumps(manifest), flush=True)


async def main(args: Args) -> None:
    for name in ("accepted", "rejected", "provisional"):
        (args.output / name).mkdir(parents=True, exist_ok=True)
    cases = [make_case(i, split, args.seed) for split, n in (("train", args.count), ("validation", args.validation_count)) for i in range(n)]
    if args.canonical_only:
        for case in cases:
            (args.output / "accepted" / f"{case['id']}.json").write_text(json.dumps(canonical_example(case)))
    else:
        client = AsyncOpenAI(api_key=args.key_file.read_text().strip(), max_retries=3, timeout=120)
        semaphore = asyncio.Semaphore(args.concurrency)
        done = 0

        async def worker(case: dict) -> None:
            nonlocal done
            async with semaphore:
                await generate_one(client, case, args.output, audit_effort=args.audit_effort, audit_tokens=args.audit_tokens)
                done += 1
                if done % 32 == 0:
                    print("PROGRESS", done, "accepted", len(list((args.output / "accepted").glob("*.json"))), flush=True)

        await asyncio.gather(*(worker(case) for case in cases))
        await client.close()
    validate_and_finalize(args)


if __name__ == "__main__":
    asyncio.run(main(Args(underscores_to_dashes=True).parse_args()))
