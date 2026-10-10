"""Larger rule-verified workflow pool with disjoint policy compositions."""

import asyncio
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from openai import AsyncOpenAI
from tap import Tap

from examples.clef.rl_pilot.generate import generate_one
from examples.clef.rl_pilot.hard import make_case as base_case
from examples.clef.rl_pilot.hard import oracle, resolve

GATES = {
    "invoice": [("split_allowed", "bank_approved"), ("verified", "bank_approved"), ("bank_approved", "split_allowed")],
    "support": [("photo_verified", "exception_signed"), ("proof", "exception_signed"), ("premium", "photo_verified")],
    "security": [("manager_signed", "device_approved"), ("device_approved", "trusted"), ("exception_signed", "manager_signed")],
    "agent": [("approval_signed", "consent"), ("verified", "approval_signed"), ("consent", "verified")],
    "tool": [("authorized", "has_id"), ("has_id", "authorized")],
    "retrieval": [("opened", "opened")],
}


class Args(Tap):
    output: Path
    key_file: Path = Path("/home/ubuntu/openai.key")
    count: int = 32768
    validation_count: int = 2048
    seed: int = 261012
    concurrency: int = 24
    canonical_only: bool = False
    audit_effort: str = "medium"
    audit_tokens: int = 8000


def make_case(index: int, split: str, seed: int) -> dict:
    case = base_case(index, "train", seed + (100000 if split == "validation" else 0))
    case["id"] = f"scaled-{split}-{index:05d}"
    case["split"] = split
    case["scenario_seed"] = f"{seed}:{split}:{index}"
    inputs = case["oracle_inputs"]
    raw = resolve(inputs["rows"], inputs["as_of"], inputs["entity"])
    effective = dict(raw)
    choices = GATES[case["family"]]
    # Train has no edits or one conjunction. Validation applies a conjunction
    # followed by an inversion of that same effective input: unseen composition.
    gate = choices[index // 8 % len(choices)]
    count = 2 if split == "validation" else index // (8 * len(choices)) % 2
    operations = []
    if count:
        target, source = gate
        effective[target] = bool(raw[target] and raw[source])
        operations.append({"kind": "and", "target": target, "source": source})
        if count == 2:
            effective[target] = not effective[target]
            operations.append({"kind": "not", "target": target})
    if operations:
        lines = ["Policy input overrides (apply AFTER resolving all ledger keys and BEFORE applying any business rule below). Apply these overrides sequentially in the stated order; they override the raw value of the named input, and all later business rules use the resulting effective values:"]
        for operation in operations:
            target = operation["target"]
            lines.append(f"Set effective {target} to the logical AND of resolved {target} and resolved {operation['source']}." if operation["kind"] == "and" else f"Replace effective {target} with its logical NOT.")
        case["policy"] = " ".join(lines) + "\n" + case["policy"]
    inputs["overrides"] = operations
    semantic = oracle(case["family"], effective, inputs["policy"])
    case["answers"] = {field: next(key for key, text in case["questions"][field]["criteria"].items() if text == answer) for field, answer in semantic.items()}
    # Distinct validation representation, preserving every ledger value.
    entity = inputs["entity"]
    first = case["facts"][0]
    rows = inputs["rows"]
    if split == "validation":
        case["facts"] = [first, "Evidence register; columns entity | key | JSON value | day | signed:\n" + "\n".join(f"{r['entity']} | {r['key']} | {json.dumps(r['value'])} | {r['day']} | {str(r['signed']).lower()}" for r in rows)]
    else:
        mode = index // 8 % 3
        if mode == 1:
            entries = [f"Register row: entity={r['entity']}; key={r['key']}; value={json.dumps(r['value'])}; day={r['day']}; signed={str(r['signed']).lower()}." for r in rows]
            case["facts"] = [first] + ["\n".join(entries[offset:offset + 8]) for offset in range(0, len(entries), 8)]
        elif mode == 2:
            case["facts"] = [first] + ["Evidence bundle: " + json.dumps(rows[offset : offset + 12], sort_keys=False) for offset in range(0, len(rows), 12)]
    template = json.dumps({"family": case["family"], "overrides": operations, "evidence_format": "table" if split == "validation" else "json-or-register"}, sort_keys=True)
    case["template_group"] = hashlib.sha256(template.encode()).hexdigest()
    payload = json.dumps({"rows": rows, "policy": inputs["policy"], "overrides": operations}, sort_keys=True).replace(entity, "CASE")
    case["scenario_group"] = hashlib.sha256(payload.encode()).hexdigest()
    case["policy_variant"] = "unseen-composition" if split == "validation" else "expanded"
    return case


def check_case(case: dict) -> None:
    inputs = case["oracle_inputs"]
    effective = resolve(inputs["rows"], inputs["as_of"], inputs["entity"])
    for op in inputs["overrides"]:
        if op["kind"] == "and":
            effective[op["target"]] = bool(effective[op["target"]] and effective[op["source"]])
        else:
            effective[op["target"]] = not effective[op["target"]]
    answers = oracle(case["family"], effective, inputs["policy"])
    assert answers == {k: case["questions"][k]["criteria"][v] for k, v in case["answers"].items()}


def finalize(args: Args) -> None:
    groups, templates, reports = {}, {}, {}
    for split, count in (("train", args.count), ("validation", args.validation_count)):
        rows = []
        for index in range(count):
            case = make_case(index, split, args.seed)
            check_case(case)
            row = json.loads((args.output / "accepted" / f"{case['id']}.json").read_text())
            assert row["metadata"]["ground_truth"] == case
            assert row["record"]["questions"] == case["questions"]
            assert case["policy"] in row["record"]["state"]
            assert all(fact in row["record"]["state"] for fact in case["facts"])
            assert row["targets"] == {k: {o: float(o == answer) for o in case["questions"][k]["criteria"]} for k, answer in case["answers"].items()}
            if not args.canonical_only:
                audit = row["metadata"]["audit"]
                assert audit["unambiguous"] is True and audit["unsupported_claims"] is False
                assert audit["answers"] == {k: case["questions"][k]["criteria"][v] for k, v in case["answers"].items()}
            row["source"] = "clef_rl_scaled_" + case["family"]
            rows.append(row)
        groups[split] = {r["metadata"]["ground_truth"]["scenario_group"] for r in rows}
        templates[split] = {r["metadata"]["ground_truth"]["template_group"] for r in rows}
        assert len(groups[split]) == count
        random.Random(f"{args.seed}:{split}").shuffle(rows)
        destination = args.output / f"{split}.jsonl"
        destination.write_text("".join(json.dumps(row) + "\n" for row in rows))
        reports[split] = {"records": count, "fields": sum(len(r["targets"]) for r in rows), "families": dict(Counter(r["source"] for r in rows)), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}
    assert not groups["train"] & groups["validation"]
    assert not templates["train"] & templates["validation"]
    manifest = {
        "seed": args.seed,
        "train": args.count,
        "validation": args.validation_count,
        "model": "none" if args.canonical_only else "gpt-6-luna",
        "report": reports,
        "sha256": {k: v["sha256"] for k, v in reports.items()},
        "checks": ["unique disjoint semantic cases", "disjoint evidence-format/policy-composition templates", "rule-derived labels", "exact evidence preservation", "canonical test fixtures only" if args.canonical_only else "blind review on every accepted case"],
        "limitations": ["Shared families and base oracle between splits", "Same-model review, no human audit", "Synthetic policy-input overrides; no organic or executed tool cases"],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.output / "validation-report.json").write_text(json.dumps(reports, indent=2))
    print("COMPLETE", json.dumps(manifest), flush=True)


async def main(args: Args) -> None:
    for name in ("accepted", "rejected", "provisional"):
        (args.output / name).mkdir(parents=True, exist_ok=True)
    client = None if args.canonical_only else AsyncOpenAI(api_key=args.key_file.read_text().strip(), max_retries=3, timeout=180)
    semaphore = asyncio.Semaphore(args.concurrency)
    done = 0

    async def worker(index: int, split: str) -> None:
        nonlocal done
        async with semaphore:
            case = make_case(index, split, args.seed)
            check_case(case)
            if args.canonical_only:
                row = {
                    "record": {"id": case["id"], "state": case["policy"] + "\n" + "\n".join(case["facts"]), "questions": case["questions"]},
                    "targets": {k: {o: float(o == v) for o in case["questions"][k]["criteria"]} for k, v in case["answers"].items()},
                    "metadata": {"ground_truth": case},
                    "source": "clef_rl_scaled_" + case["family"],
                }
                (args.output / "accepted" / f"{case['id']}.json").write_text(json.dumps(row))
            else:
                await generate_one(client, case, args.output, audit_effort=args.audit_effort, audit_tokens=args.audit_tokens)
            done += 1
            if done % 64 == 0:
                print("PROGRESS", done, flush=True)

    try:
        await asyncio.gather(*(worker(i, split) for split, count in (("train", args.count), ("validation", args.validation_count)) for i in range(count)))
    finally:
        if client is not None:
            await client.close()
    finalize(args)


if __name__ == "__main__":
    asyncio.run(main(Args(underscores_to_dashes=True).parse_args()))
