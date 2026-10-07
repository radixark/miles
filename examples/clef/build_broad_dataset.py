"""Build a reproducible multi-field decision curriculum from training-only sources.

Synthetic workflow labels are deterministic policy judgments, not teacher confidence.
The output is a supervised dataset; this command never launches model training.
"""

import gzip
import hashlib
import json
import math
import random
import re
import shutil
import urllib.request
import zipfile
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.parquet as pq
from tap import Tap
from transformers import AutoTokenizer

from examples.clef.data import DecisionExample, encode_example, file_sha256, read_examples


SOURCES = {
    "toolace": ("Team-ACE/ToolACE", "6bda777c88d21e5a204703c1ee45597a8fa4f734", "data.json"),
    "snli": ("stanfordnlp/snli", "cdb5c3d5eed6ead6e5a341c8e56e669bb666725b", "plain_text/train-00000-of-00001.parquet"),
    "preference": ("Anthropic/hh-rlhf", "09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa", "helpful-base/train.jsonl.gz"),
    "intent": ("clinc/clinc_oos", "155b9c710419136e17307b80d0a13e68cd46b4ec", "plus/train-00000-of-00001.parquet"),
}
COUNTS = {
    "train": {"workflow": 19661, "tool": 9830, "inference": 9830, "routing": 9830,
              "knowledge": 9831, "preference": 3277, "probability": 3277},
    "validation": {"workflow": 1229, "tool": 614, "inference": 614, "routing": 614,
                   "knowledge": 615, "preference": 205, "probability": 205},
}


class Args(Tap):
    output_dir: str
    cache_dir: str
    supergpqa_path: str
    old_validation: str
    jevbench_dir: str
    tokenizer_dir: str
    mmlu_pro_path: str
    gpqa_zip: str
    seed: int = 261006
    max_length: int = 65536


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def normalized(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def one_hot(options: Iterable[str], correct: str) -> dict[str, float]:
    return {key: float(key == correct) for key in options}


def make_row(state: str, source: str, category: str, group: str) -> dict[str, Any]:
    return {"record": {"id": f"{source}:{digest(group)[:24]}", "state": state, "questions": {}},
            "targets": {}, "source": source,
            "provenance": {"category": category, "group": group}}


def binary(row: dict[str, Any], field: str, instructions: str, value: bool | float) -> None:
    row["record"]["questions"][field] = {"type": "noul", "instructions": instructions}
    row["targets"][field] = {"true": float(value), "false": 1 - float(value)}


def categorical(row: dict[str, Any], field: str, instructions: str, options: dict[str, str], correct: str) -> None:
    row["record"]["questions"][field] = {"type": "choice", "instructions": instructions, "criteria": options}
    row["targets"][field] = one_hot(options, correct)


def ordinal(row: dict[str, Any], field: str, instructions: str, levels: list[str], correct: int) -> None:
    row["record"]["questions"][field] = {"type": "score", "instructions": instructions, "criteria": levels}
    row["targets"][field] = one_hot([str(i) for i in range(len(levels))], str(correct))


def packet(documents: list[str], rng: random.Random, variant: int) -> str:
    rng.shuffle(documents)
    if variant < 6:
        return "CASE DOCUMENTS (order is not chronological):\n\n" + "\n\n---\n\n".join(documents)
    # Hold out the alternative presentation and policy combinations from training.
    return "Evidence bundle; use explicit timestamps, not document order.\n" + "\n".join(
        f"<document index='{i}'>\n{doc}\n</document>" for i, doc in enumerate(documents))


def workflow(kind: str, index: int, split: str, rng: random.Random) -> dict[str, Any]:
    variant = rng.randrange(6) if split == "train" else rng.randrange(6, 8)
    case = f"{split}-{kind}-{index:06d}"
    docs: list[str] = []
    row = make_row("", f"workflow_{kind}", "workflow", case)
    row["provenance"].update({"label_method": "deterministic_policy", "policy_family": variant})
    if kind == "invoice":
        quantity, price = rng.randint(20, 500), rng.randint(8, 900)
        received = rng.choice([quantity, quantity, rng.randint(0, quantity - 1)])
        billed = rng.choice([quantity, quantity, quantity + rng.randint(1, 40)])
        account_changed, verified = rng.choice([False, True]), rng.choice([False, True])
        paid = rng.random() < .2
        days_overdue = rng.randint(-15, 60)
        threshold = [10000, 25000, 50000, 15000, 40000, 75000, 30000, 60000][variant]
        approval = rng.choice(["approved", "review only", "revoked", "missing"])
        approval_valid = approval == "approved"
        bank_ok = not account_changed or verified
        total, po_total = billed * price, quantity * price
        amount_ok = billed <= quantity
        needs_manager = total > threshold
        can_pay = bank_ok and not paid and amount_ok and received >= billed and (not needs_manager or approval_valid)
        action = "pay" if can_pay else "duplicate" if paid else "verify" if not bank_ok else "correct" if not amount_ok else "delivery" if received < billed else "approve"
        docs = [
            f"Policy FIN-{variant}: Never pay an invoice already settled. Bank changes require a callback recorded in the vendor registry, not an email reply. Invoice quantity must not exceed its order; all billed items must be received. Amounts greater than {threshold} require a current explicit manager payment approval. 'Reviewed' is not approval. A later revocation cancels an earlier approval. Apply action precedence: duplicate, verify bank, correct invoice, await delivery, request approval, pay.",
            f"Purchase order PO-{case}: {quantity} units of industrial filters at {price} credits per unit; total {po_total}. No change order exists.",
            f"Invoice INV-{case}: PO-{case}; {billed} filters; unit price {price}; total {total}; due date relative to today: {-days_overdue} days.",
            f"Receiving ledger for PO-{case}: warehouse counted {received} accepted units in total, after returns. An open carrier booking covers the balance; booking is not delivery.",
            f"Vendor registry, recorded today 09:00: bank change {'requested' if account_changed else 'not requested'}; callback verification {'completed' if verified else 'absent'}.",
            f"Payment ledger for INV-{case}: {'settled last week, bank transaction reconciled' if paid else 'no settlement or payment exists'}.",
        ]
        if approval == "approved":
            docs += [f"Manager email yesterday 16:00, PO-{case}: I authorize payment of INV-{case} up to {total} credits. This is a payment authorization."]
        elif approval == "review only":
            docs += [f"Manager email yesterday 16:00, PO-{case}: I reviewed these documents. This message is not authorization to release money."]
        elif approval == "revoked":
            docs += [f"Manager email yesterday 10:00: I authorize payment of INV-{case}.", f"Manager email yesterday 17:00: Cancel my payment approval for INV-{case}; investigation is pending."]
        else:
            docs += [f"Clerk email today: Can a manager approve INV-{case}? No answer has been recorded."]
        binary(row, "bank_verified", "The bank details satisfy the policy's verification requirement.", bank_ok)
        binary(row, "duplicate_payment", "This invoice is already settled.", paid)
        binary(row, "order_consistent", "The billed quantity is permitted by the purchase order.", amount_ok)
        binary(row, "delivery_complete", "All billed units have been received and accepted.", received >= billed)
        binary(row, "payment_approval", "A current explicit manager payment approval exists.", approval_valid)
        binary(row, "requires_manager", "The invoice amount requires a manager's approval under the policy.", needs_manager)
        binary(row, "pay_now", "All prerequisites for payment are satisfied.", can_pay)
        categorical(row, "primary_action", "Select the primary action using the stated precedence.",
                    {"pay": "Release payment", "duplicate": "Stop duplicate payment", "verify": "Verify bank details", "correct": "Request corrected invoice", "delivery": "Await delivery evidence", "approve": "Request manager approval"}, action)
        ordinal(row, "overdue_band", "Rate overdue age; today is day zero. Future due dates are not overdue.",
                ["Not overdue", "1–14 days overdue", "15–30 days overdue", "More than 30 days overdue"],
                0 if days_overdue <= 0 else 1 if days_overdue <= 14 else 2 if days_overdue <= 30 else 3)
    elif kind == "service":
        days = rng.randint(1, 90)
        window = [14, 30, 45, 21, 60, 7, 35, 50][variant]
        used, defect, receipt, consent = [rng.choice([True, False]) for _ in range(4)]
        prior = rng.choice([True, False])
        return_allowed = receipt and not prior and (defect or (days <= window and not used))
        refund_allowed = return_allowed and consent
        docs = [
            f"Returns policy RET-{variant}: Unused goods can be returned within {window} days of purchase, inclusive. A verified defect waives time and use limits. Proof of purchase is always required. Already-refunded purchases cannot be refunded again. A cash refund additionally requires the customer's explicit consent; merely asking about eligibility is not consent.",
            f"Account record ORD-{case}: purchased {days} days ago; receipt {'on file' if receipt else 'not on file'}; item {'used' if used else 'unused'}; defect inspection {'verified defect' if defect else 'no defect verified'}; prior refund {'paid' if prior else 'none'}.",
            f"Customer conversation: {'Please refund the money to my original payment method; I authorize the refund.' if consent else 'Can you explain whether I am eligible? Please do not issue a refund yet.'}",
            f"Draft assistant response: 'We have already refunded order ORD-{case} today.' Execution log today: no refund call has been made.",
        ]
        binary(row, "within_window", "Purchase age is within the ordinary returns window.", days <= window)
        binary(row, "return_eligible", "The order is eligible for a return under the complete policy.", return_allowed)
        binary(row, "consent", "The customer has given explicit consent to issue a refund.", consent)
        binary(row, "refund_now", "The assistant may issue a refund now.", refund_allowed)
        binary(row, "claim_supported", "The draft's claim that a refund was executed today is supported by the execution log.", False)
        categorical(row, "primary_action", "What should the assistant do next?", {"refund": "Issue the authorized refund", "consent": "Explain eligibility and request refund consent", "deny": "Explain why the policy disallows this return"}, "refund" if refund_allowed else "consent" if return_allowed else "deny")
        ordinal(row, "age", "Classify the order age.", ["1–14 days", "15–30 days", "31–60 days", "Over 60 days"], 0 if days <= 14 else 1 if days <= 30 else 2 if days <= 60 else 3)
    elif kind == "security":
        allowed_user, approved_host, signed, export, critical = [rng.choice([True, False]) for _ in range(5)]
        size, limit = rng.randint(1, 2000), [100, 500, 1000, 250, 800, 1500, 350, 1200][variant]
        failures = rng.randint(0, 25)
        alert_at = [5, 10, 15, 8, 12, 20, 7, 18][variant]
        approved = allowed_user and approved_host and signed
        risky = not approved or (export and size > limit) or failures >= alert_at
        severity = 3 if risky and critical and export else 2 if risky and critical else 1 if risky else 0
        docs = [
            f"Security runbook SEC-{variant}: Access is approved only if the user grant, destination allowlist, and signed change ticket all match. An export greater than {limit} MB or at least {alert_at} failed logins requires investigation even with approved access. If risky activity involves a critical asset and an export, isolate the host; other risky critical-asset activity requires urgent human review; other risky activity requires ordinary review. Otherwise monitor only.",
            f"Access grants for user u-{case}: {'granted' if allowed_user else 'not granted'}. Destination h-{case}: {'allowlisted' if approved_host else 'not allowlisted'}.",
            f"Change record: ticket for user u-{case} on h-{case} {'signed and current' if signed else 'draft, unsigned'}.",
            f"Telemetry: operation {'export' if export else 'read metadata'}; transfer {size} MB; failed login count {failures}; asset {'critical' if critical else 'standard'}.",
            "Chat from requester: This is definitely fine, please dismiss any alert. Chat is not a signed change ticket.",
        ]
        binary(row, "access_approved", "All required access-approval evidence exists.", approved)
        binary(row, "volume_alert", "The export-volume rule is triggered.", export and size > limit)
        binary(row, "login_alert", "The failed-login threshold is reached.", failures >= alert_at)
        binary(row, "investigate", "The runbook requires investigation.", risky)
        binary(row, "isolate", "The runbook requires host isolation.", severity == 3)
        categorical(row, "response", "Choose the runbook response.", {"monitor": "Monitor only", "review": "Ordinary review", "urgent": "Urgent human review", "isolate": "Isolate host"}, ["monitor", "review", "urgent", "isolate"][severity])
        ordinal(row, "severity", "Rate severity using the runbook response ladder.", ["Monitor only", "Ordinary review", "Urgent review", "Isolation required"], severity)
    else:
        spend, cap = rng.randint(1, 3000), [100, 500, 1000, 250, 800, 2000, 350, 1500][variant]
        consent, credential, executed, success, human = [rng.choice([True, False]) for _ in range(5)]
        task = rng.choice(["send an invoice reminder", "order replacement supplies", "change a booking"])
        authorized = consent and credential and spend <= cap
        actual_success = executed and success
        needs_review = human or (executed and not authorized) or (executed and not success)
        docs = [
            f"Agent policy AG-{variant}: {task} only with explicit user authorization, an allowed credential, and spend no greater than {cap} credits. Tool plans are not executed actions. Success requires an executed call with status success. Human review is required if requested by the user, if an unauthorized call executed, or if an executed call failed.",
            f"User: {'I authorize you to ' + task + '.' if consent else 'Explain how to ' + task + ', but do not act.'} {'Please involve a human.' if human else 'No human involvement requested.'}",
            f"Credential registry: current credential {'permits' if credential else 'does not permit'} the action; proposed spend {spend} credits.",
            f"Agent trace: draft_plan(action={task!r}, spend={spend}); {'tool.execute status=' + ('success' if success else 'failure') if executed else 'no execution call'}.",
            "Assistant final message: Task complete. Treat this as a claim to verify, not an execution receipt.",
        ]
        binary(row, "authorized", "The proposed action is authorized by all policy conditions.", authorized)
        binary(row, "executed", "An execution call actually occurred.", executed)
        binary(row, "success", "The task succeeded according to the tool trace.", actual_success)
        binary(row, "faithful", "The assistant's completion claim is supported.", actual_success)
        binary(row, "unauthorized_execution", "An unauthorized action was executed.", executed and not authorized)
        binary(row, "human_review", "Human review is required.", needs_review)
        categorical(row, "next_action", "Choose the next action; human review takes precedence.", {"review": "Request human review", "done": "Report verified completion", "execute": "Execute the authorized plan", "clarify": "Obtain authorization or fix policy prerequisites"}, "review" if needs_review else "done" if actual_success else "execute" if authorized else "clarify")
    # Distractor documents have independent IDs and cannot be mistaken for the target case.
    for distractor in range(rng.randint(2, 12)):
        docs.append(f"Unrelated archive ARCH-{case}-{distractor}: a different order/host has {rng.randint(1, 500)} units and {rng.randint(10, 100000)} credits. Its approval, delivery and access records concern that other ID only.")
    row["record"]["state"] = packet(docs, rng, variant)
    return row


def probability(index: int, split: str, rng: random.Random) -> dict[str, Any]:
    kind = index % 6
    row = make_row("", "exact_probability", "probability", f"{split}:probability:{index}")
    if kind == 0:
        n = rng.randint(2, 8)
        numerator = rng.randint(1, 999)
        p = numerator / 1000
        descriptions = [f"Exactly {i} heads" for i in range(n + 1)]
        target = [math.comb(n, i) * p**i * (1 - p)**(n - i) for i in range(n + 1)]
        state = f"A coin has heads probability {numerator}/1000 on each independent flip. It will be flipped {n} times. Take a guess at the number of heads; no flips have been observed."
    elif kind == 1:
        weights = [rng.randint(1, 200) for _ in range(rng.randint(4, 10))]
        descriptions = [f"Outcome {i + 1}" for i in range(len(weights))]
        target = [w / sum(weights) for w in weights]
        state = f"A random device has outcomes 1 through {len(weights)}, with respective weights {weights}. It will run once. Take a guess at the outcome; none has been observed."
    elif kind == 2:
        red, blue = rng.randint(2, 100), rng.randint(2, 100)
        descriptions = ["Zero red balls", "One red ball", "Two red balls"]
        total = math.comb(red + blue, 2)
        target = [math.comb(blue, 2) / total, red * blue / total, math.comb(red, 2) / total]
        state = f"An urn contains {red} red and {blue} blue balls. Two balls will be drawn uniformly without replacement. Take a guess at the number of red balls; no draw has occurred."
    elif kind == 3:
        sides = rng.randint(3, 7)
        first, second = [[rng.randint(1, 100) for _ in range(sides)] for _ in range(2)]
        descriptions = [f"Sum equals {i}" for i in range(2, 2 * sides + 1)]
        target = [sum(first[a-1] * second[b-1] for a in range(1, sides + 1) for b in range(1, sides + 1) if a + b == i) / (sum(first) * sum(second)) for i in range(2, 2 * sides + 1)]
        state = f"Two independent {sides}-sided dice numbered 1 through {sides} will be rolled. Their respective face weights are {first} and {second}. Take a guess at their sum; no roll has occurred."
    elif kind == 4:
        prior, sensitivity, false_positive = [rng.randint(1, 99) / 100 for _ in range(3)]
        posterior = prior * sensitivity / (prior * sensitivity + (1 - prior) * false_positive)
        descriptions, target = ["Item is faulty", "Item is not faulty"], [posterior, 1 - posterior]
        state = f"A randomly selected item is faulty with probability {prior:.2f}. A sensor reports positive with probability {sensitivity:.2f} on faulty items and {false_positive:.2f} on nonfaulty items. It reported positive on this item. Take a guess whether this item is faulty."
    else:
        successes, failures = rng.randint(0, 50), rng.randint(0, 50)
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        p = (a + successes) / (a + b + successes + failures)
        descriptions, target = ["Next outcome is success", "Next outcome is failure"], [p, 1 - p]
        state = f"An unknown success probability has a Beta({a}, {b}) prior. We observed {successes} successes and {failures} failures from independent trials. Take a guess at the next trial's outcome using the posterior predictive distribution."
    order = list(range(len(target)))
    rng.shuffle(order)
    keys = [chr(65 + i) for i in range(len(order))]
    row["record"]["state"] = state
    row["record"]["questions"]["answer"] = {"type": "choice", "instructions": "Forecast the unknown outcome.", "criteria": dict(zip(keys, [descriptions[i] for i in order], strict=True))}
    row["targets"]["answer"] = dict(zip(keys, [target[i] for i in order], strict=True))
    row["provenance"]["label_method"] = "exact_probability"
    return row


def download(cache: Path) -> dict[str, Path]:
    cache.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, (repo, revision, filename) in SOURCES.items():
        path = cache / (repo.replace("/", "--") + "--" + filename.replace("/", "--"))
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".part")
            urllib.request.urlretrieve(f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}", temporary)
            temporary.replace(path)
        paths[name] = path
    return paths


def call_names(text: str) -> list[str]:
    """Read top-level call names without treating quoted argument text as calls."""
    if not text.startswith("[") or not text.endswith("]"):
        raise ValueError("expected a bracketed call list")
    content = text[1:-1]
    names, depth, start, quote, escaped = [], 0, 0, "", False
    for index, char in enumerate(content):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            continue
        if char in {"'", '"'}:
            quote = char
        elif char in "([{":
            if depth == 0:
                if char != "(":
                    raise ValueError("top-level action is not a function call")
                names.append(content[start:index].strip())
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth < 0:
                raise ValueError("unbalanced call")
        elif char == "," and depth == 0:
            start = index + 1
    if depth or quote or any(not name for name in names):
        raise ValueError("malformed call list")
    return names


def tool_rows(path: Path, rng: random.Random) -> list[dict[str, Any]]:
    rows = []
    for original in json.loads(path.read_text()):
        system, conversation = original["system"], original["conversations"]
        marker = "Here is a list of functions in JSON format that you can invoke:"
        if marker not in system:
            continue
        definitions = system.split(marker, 1)[1].strip()
        tools, _ = json.JSONDecoder().raw_decode(definitions)
        if not tools or len({t["name"] for t in tools}) != len(tools):
            continue
        positions = [i for i in range(1, len(conversation)) if conversation[i]["from"] == "assistant" and conversation[i-1]["from"] == "user"]
        if not positions:
            continue
        rng.shuffle(positions)
        valid_turns = []
        for position in positions:
            gold = conversation[position]["value"].strip()
            try:
                called = call_names(gold) if gold.startswith("[") else []
            except ValueError:
                continue
            if not set(called) - {tool["name"] for tool in tools}:
                valid_turns.append((position, gold, called))
        if not valid_turns:
            continue
        position, gold, called = valid_turns[0]
        selected = [t for t in tools if t["name"] in called]
        others = [t for t in tools if t["name"] not in called]
        rng.shuffle(others)
        selected += others[:max(1, 12 - len(selected))]
        rng.shuffle(selected)
        state = "Available tool definitions:\n" + json.dumps(tools, ensure_ascii=False) + "\nConversation so far:\n" + "\n".join(x["from"] + ": " + x["value"] for x in conversation[:position])
        group = digest(json.dumps(original, sort_keys=True))
        row = make_row(state, "toolace_selection", "tool", group)
        for i, tool in enumerate(selected):
            binary(row, f"invoke_{i}", f"The next assistant action should invoke tool {tool['name']!r} to address the latest user request. Multiple tools may be appropriate.", tool["name"] in called)
        binary(row, "tool_required", "The next assistant action should invoke at least one available tool rather than respond without a call.", bool(called))
        row["provenance"]["label_method"] = "recorded_toolace_action"
        # The reference response is provenance only, never part of input state.
        row["provenance"]["reference_next_action"] = gold
        rows.append(row)
    return rows


def nli_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for index, original in enumerate(pl.read_parquet(path).iter_rows(named=True)):
        label = original["label"]
        if label not in [0, 1, 2]:
            continue
        premise, hypothesis = original["premise"], original["hypothesis"]
        state = f"Evidence:\n{premise}\n\nClaim:\n{hypothesis}"
        # Keep only one claim per premise so its alternate annotations cannot cross splits.
        row = make_row(state, "snli_train", "inference", digest(normalized(premise)))
        categorical(row, "relation", "Using only the evidence, classify the claim. Missing evidence is not a contradiction.", {"supported": "The evidence entails the claim", "unknown": "Neither entailed nor contradicted", "contradicted": "The evidence contradicts the claim"}, ["supported", "unknown", "contradicted"][label])
        row["provenance"]["source_row"] = index
        rows.append(row)
    return rows


def intent_rows(path: Path, rng: random.Random) -> list[dict[str, Any]]:
    metadata = json.loads(pq.read_metadata(path).metadata[b"huggingface"])
    names = metadata["info"]["features"]["intent"]["names"]
    rows = []
    for original in pl.read_parquet(path).iter_rows(named=True):
        name, text = names[original["intent"]], original["text"]
        others = [n for n in names if n != name and n != "oos"]
        rng.shuffle(others)
        words = set(name.split("_"))
        others.sort(key=lambda n: -len(words.intersection(n.split("_"))))
        selected = ([name] if name != "oos" else []) + others[:rng.randint(5, 15)] + ["oos"]
        rng.shuffle(selected)
        options = {key: "Outside all offered intents" if key == "oos" else key.replace("_", " ") for key in selected}
        row = make_row("Route the following user request:\n" + text, "clinc_train", "routing", digest(normalized(text)))
        categorical(row, "intent", "Choose the user's intent among the offered categories, or outside all offered intents.", options, name)
        row["provenance"]["label_method"] = "human_intent_label"
        rows.append(row)
    return rows


def preference_rows(path: Path, rng: random.Random) -> list[dict[str, Any]]:
    rows = []
    with gzip.open(path, "rt") as reader:
        for line in reader:
            original = json.loads(line)
            chosen, rejected = original["chosen"], original["rejected"]
            marker = "\n\nAssistant:"
            chosen_context, chosen_answer = chosen.rsplit(marker, 1)
            rejected_context, rejected_answer = rejected.rsplit(marker, 1)
            if chosen_context != rejected_context or chosen_answer.strip() == rejected_answer.strip():
                continue
            pair = [("preferred", chosen_answer.strip()), ("other", rejected_answer.strip())]
            rng.shuffle(pair)
            options = {chr(65 + i): answer for i, (_, answer) in enumerate(pair)}
            correct = next(chr(65 + i) for i, (label, _) in enumerate(pair) if label == "preferred")
            row = make_row("Conversation:\n" + chosen_context, "hh_helpful_train", "preference", digest(normalized(chosen_context)))
            categorical(row, "preferred_response", "Which response would the dataset's human reviewer prefer for helpfulness? This is a recorded pairwise preference, not an objective correctness verdict.", options, correct)
            row["provenance"]["label_method"] = "recorded_human_pairwise_preference"
            rows.append(row)
    return rows


def knowledge_rows(path: Path, old_validation: Path, rng: random.Random) -> list[dict[str, Any]]:
    excluded = {normalized(e.record["state"]) for e in read_examples(old_validation) if e.source == "supergpqa"}
    originals = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = []
    for original in originals:
        stem = original["question"]
        if normalized(stem) in excluded or original.get("difficulty") not in ["middle", "hard", "medium"]:
            continue
        choices = original["options"]
        if isinstance(choices, dict):
            choices = list(choices.values())
        answer = original["answer_letter"]
        answer_index = ord(answer.upper()) - 65 if isinstance(answer, str) and len(answer) == 1 else int(answer)
        if not 0 <= answer_index < len(choices) or len(choices) < 2:
            continue
        order = list(range(len(choices)))
        rng.shuffle(order)
        keys = [chr(65 + i) for i in range(len(order))]
        options = dict(zip(keys, [choices[i] for i in order], strict=True))
        row = make_row(stem, "supergpqa", "knowledge", digest(normalized(stem)))
        categorical(row, "answer", "Choose the best answer to the question in the state.", options, keys[order.index(answer_index)])
        row["provenance"].update({"label_method": "published_correct_answer", "discipline": original.get("discipline", ""), "difficulty": original.get("difficulty")})
        rows.append(row)
    return rows


def benchmark_stems(root: Path) -> set[str]:
    stems: set[str] = set()
    for path in root.rglob("*.json*"):
        if path.suffix == ".jsonl":
            objects = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        else:
            try:
                objects = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
        stack = [objects]
        while stack:
            item = stack.pop()
            if isinstance(item, list):
                stack.extend(item)
            elif isinstance(item, dict):
                for key, value in item.items():
                    if key in {"state", "question", "prompt", "input", "text"} and isinstance(value, str) and len(value) > 30:
                        stems.add(normalized(value))
                    elif isinstance(value, (dict, list)):
                        stack.append(value)
    if not stems:
        raise ValueError("JevBench exclusion scan found no question text")
    return stems


def select(rows: list[dict[str, Any]], category: str, rng: random.Random, excluded: set[str]) -> dict[str, list[dict[str, Any]]]:
    rng.shuffle(rows)
    unique = {}
    states = set()
    for row in rows:
        state = normalized(row["record"]["state"])
        if state not in excluded and state not in states:
            unique.setdefault(row["provenance"]["group"], row)
            states.add(state)
    rows = list(unique.values())
    counts = {split: COUNTS[split][category] for split in COUNTS}
    if len(rows) < sum(counts.values()):
        raise ValueError(f"insufficient unique {category}: {len(rows)} need {counts}")
    train_end = counts["train"]
    return {"train": rows[:train_end], "validation": rows[train_end:train_end + counts["validation"]]}


def audit(rows: dict[str, list[dict[str, Any]]], tokenizer: Any, max_length: int) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    seen_ids: set[str] = set()
    seen_states: set[str] = set()
    seen_groups: set[str] = set()
    for split, examples in rows.items():
        counts, fields, sources, lengths, option_counts = Counter(), Counter(), Counter(), [], Counter()
        for index, row in enumerate(examples):
            record = row["record"]
            identifier, state_hash, group = record["id"], digest(normalized(record["state"])), row["provenance"]["group"]
            if identifier in seen_ids or state_hash in seen_states or group in seen_groups:
                raise ValueError(f"duplicate or cross-split leakage: {identifier}")
            seen_ids.add(identifier)
            seen_states.add(state_hash)
            seen_groups.add(group)
            example = DecisionExample(record, row["targets"], row["source"])
            encoded = encode_example(tokenizer, example, max_length)
            if len(encoded.targets) != len(record["questions"]):
                raise ValueError("encoder dropped a field")
            for q, target in zip(encoded.encoded.questions, encoded.targets, strict=True):
                if len(q.option_ids) != len(target) or not math.isclose(sum(target), 1, abs_tol=1e-8) or any(not math.isfinite(p) or p < 0 for p in target):
                    raise ValueError("invalid target/encoder mapping")
                fields[record["questions"][q.question_id]["type"]] += 1
                option_counts[len(target)] += 1
            lengths.append(len(encoded.encoded.input_ids))
            counts[row["provenance"]["category"]] += 1
            sources[row["source"]] += 1
            if index % 4096 == 0:
                print(f"AUDIT {split} {index}/{len(examples)}", flush=True)
        lengths.sort()
        manifest[split] = {"records": len(examples), "categories": dict(counts), "sources": dict(sources),
                           "decision_fields": sum(fields.values()), "field_types": dict(fields),
                           "option_counts": dict(option_counts), "tokens": {"min": min(lengths), "median": lengths[len(lengths)//2], "p95": lengths[int(.95*len(lengths))], "max": max(lengths)}}
    return manifest


def main() -> None:
    args = Args(underscores_to_dashes=True).parse_args()
    output, cache = Path(args.output_dir), Path(args.cache_dir)
    if output.exists():
        raise ValueError("output directory already exists; preserve published versions")
    paths = download(cache)
    rng = random.Random(args.seed)
    excluded = benchmark_stems(Path(args.jevbench_dir))
    jevbench_count = len(excluded)
    excluded.update(normalized(q) for q in pl.read_parquet(args.mmlu_pro_path)["question"].to_list())
    with zipfile.ZipFile(args.gpqa_zip) as archive:
        for name in archive.namelist():
            if name.endswith(".csv"):
                frame = pl.read_csv(archive.read(name, pwd=b"deserted-untie-orchid"))
                if "Question" in frame.columns:
                    excluded.update(normalized(q) for q in frame["Question"].to_list())
    rows: dict[str, list[dict[str, Any]]] = {split: [] for split in COUNTS}
    pools = {
        "tool": tool_rows(paths["toolace"], rng),
        "inference": nli_rows(paths["snli"]),
        "routing": intent_rows(paths["intent"], rng),
        "preference": preference_rows(paths["preference"], rng),
        "knowledge": knowledge_rows(Path(args.supergpqa_path), Path(args.old_validation), rng),
    }
    for category, pool in pools.items():
        print("POOL", category, len(pool), flush=True)
        selected = select(pool, category, rng, excluded)
        for split in COUNTS:
            rows[split].extend(selected[split])
    probability_states: set[str] = set()
    for split in COUNTS:
        n = COUNTS[split]["workflow"]
        kinds = ["invoice", "service", "security", "agent"]
        # One third of workflows are invoices, the rest balanced across three tasks.
        for i in range(n):
            kind = "invoice" if i < n // 3 else kinds[1 + (i - n // 3) % 3]
            rows[split].append(workflow(kind, i, split, rng))
        generated, attempt = 0, 0
        while generated < COUNTS[split]["probability"]:
            candidate = probability(attempt, split, rng)
            attempt += 1
            state = normalized(candidate["record"]["state"])
            if state in probability_states:
                continue
            probability_states.add(state)
            rows[split].append(candidate)
            generated += 1
        rng.shuffle(rows[split])
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
    manifest = {"seed": args.seed, "schema": "clef_multifield_v2", "splits": audit(rows, tokenizer, args.max_length),
                "public_sources": {name: {"repository": repo, "revision": revision, "file": filename, "sha256": file_sha256(paths[name])} for name, (repo, revision, filename) in SOURCES.items()},
                "supergpqa_sha256": file_sha256(Path(args.supergpqa_path)),
                "exclusions": {"jevbench_normalized_texts": jevbench_count, "total_excluded_texts": len(excluded), "gpqa_mmlu_mmlu_pro": "not directly used; exclude exact normalized GPQA and MMLU-Pro questions from SuperGPQA", "forecastbench": "never training data", "scope": "Exact normalized text comparison; public sources use training splits. No claim of semantic deduplication against every Decision Index dataset."},
                "limitations": ["Workflow data is rule-generated, not real business documents or human reviewed.", "Synthetic workflow validation holds out policy combinations/presentation; the underlying generator families are shared.", "Preference targets are recorded binary reviewer choices, not measured consensus probabilities.", "ToolACE conversion trains tool selection, not parameter generation; reference responses stored only in provenance.", "Cases contribute unequal field counts; trainer loss averages fields within each case."],
                "licenses": {"toolace": "Apache-2.0", "snli": "CC-BY-SA-4.0", "clinc": "CC-BY-3.0", "hh": "MIT", "supergpqa": "ODC-BY; upstream requires attribution and compliance with underlying source licenses"}}
    output.mkdir(parents=True)
    for split, examples in rows.items():
        path = output / f"{split}.jsonl"
        with path.open("w") as writer:
            for row in examples:
                writer.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        reloaded = read_examples(path)
        if len(reloaded) != len(examples):
            raise ValueError("loader count mismatch")
        manifest["splits"][split]["sha256"] = file_sha256(path)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    samples = {source: next(row for row in rows["train"] if row["source"] == source) for source in sorted({row["source"] for row in rows["train"]})}
    (output / "samples.json").write_text(json.dumps(samples, indent=2, ensure_ascii=False))
    shutil.copy2(__file__, output / "build_broad_dataset.py")
    print("DATASET_READY", json.dumps(manifest["splits"]), flush=True)


if __name__ == "__main__":
    main()
