"""Report the end-to-end benchmark after exact selected-route oracle comparison."""

import json
import math
from pathlib import Path


def _read(path):
    return json.loads(path.read_text())


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _seconds(value):
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid timing: {value}")
    return value


def _generations(records, routes, version=None):
    selected = {}
    for record in records:
        rank = record["dp_rank"]
        for engine, response in zip(record["engine_ids"], record["engines"], strict=True):
            route = (engine, rank)
            meta, tokens = response["meta_info"], response["output_ids"]
            if route in selected or meta["dp_rank"] != rank:
                raise ValueError("Duplicate or incorrectly routed generation")
            output = meta["output_token_logprobs"]
            if not tokens or len(tokens) != meta["completion_tokens"] or len(output) != len(tokens):
                raise ValueError("Incomplete generation output/logprobs")
            if any(row[1] != token or not math.isfinite(row[0]) for row, token in zip(output, tokens, strict=True)):
                raise ValueError("Unaligned or nonfinite output logprobs")
            inputs = meta["input_token_logprobs"]
            if len(inputs) != meta["prompt_tokens"]:
                raise ValueError("Incomplete prompt logprobs; request logprob_start_len=0")
            if any(not (i == 0 and row[0] is None) and not math.isfinite(row[0]) for i, row in enumerate(inputs)):
                raise ValueError("Nonfinite input logprobs")
            actual_version = str(meta["weight_version"])
            if version is not None and actual_version != str(version):
                raise ValueError("Generation used the wrong weight version")
            end = 0
            for span in meta["weight_versions"]:
                if (
                    span["start"] != end
                    or not end < span["end"] <= len(tokens)
                    or str(span["version"]) != actual_version
                ):
                    raise ValueError("Mixed or incomplete generation version spans")
                end = span["end"]
            if end != len(tokens) or meta["prompt_tokens"] <= 0:
                raise ValueError("Incomplete generation version/prompt coverage")
            selected[route] = {
                "text": response["text"],
                "output_ids": tokens,
                "prompt_tokens": meta["prompt_tokens"],
                "input_token_logprobs": inputs,
                "output_token_logprobs": output,
            }
    if set(selected) != routes:
        raise ValueError("Missing or unexpected generation routes")
    return selected


def _compare(left, right):
    rows = []
    for engine, rank in sorted(left):
        a, b = left[engine, rank], right[engine, rank]
        equal = {key: a[key] == b[key] for key in a}
        rows.append({"engine_id": engine, "dp_rank": rank, "equal": equal, "exact": all(equal.values())})
    return {"status": "PASS" if all(row["exact"] for row in rows) else "FAILED", "routes": rows}


def _rank_rows(update, publication, identities):
    receipt = update["receipt"]
    stages = []
    for key, state in (("receipts", "APPLIED"), ("resumed_receipts", "RESUMED")):
        by_rank = {row["identity"]["rank_id"]: row for row in receipt[key]}
        if len(by_rank) != len(receipt[key]) or set(by_rank) != set(identities):
            raise ValueError("Missing or duplicate rank receipts")
        for rank, row in by_rank.items():
            if row["identity"] != identities[rank] or row["state"] != state or not row["result"]["applied"]:
                raise ValueError("Rank identity or apply/resume state differs")
            if row["session_id"] != receipt["session_id"]:
                raise ValueError("Rank session differs")
            for field in ("manifest_sha256", "stream_id", "base_version", "target_version", "plan_digest"):
                if row[field] != publication[field]:
                    raise ValueError(f"Rank publication differs: {field}")
            if row["result"]["target_version"] != publication["target_version"]:
                raise ValueError("Applied target differs")
        stages.append(by_rank)
    rows = []
    for rank, resumed in stages[1].items():
        timing = resumed["scheduler_timing"]
        start, fence, end = (timing[key] for key in ("pause_started_ns", "reader_fence_completed_ns", "resumed_ns"))
        if resumed["generation_paused"] or end is None or not start <= fence <= end:
            raise ValueError("Scheduler pause did not complete")
        pause = _seconds(timing["blocked_s"])
        if not math.isclose(pause, (end - start) / 1e9, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("Scheduler pause endpoints differ from reported duration")
        if start != stages[0][rank]["scheduler_timing"]["pause_started_ns"]:
            raise ValueError("Apply and resume pause endpoints differ")
        result, metrics = resumed["result"], resumed["result"]["timings"]
        cuda = metrics["cuda_event_ms"] if result["timing_enabled"] else None
        rows.append(
            {
                "identity": identities[rank],
                "outer_cpu_s": _seconds(metrics["host_rank_outer_zstd_decode_s"]),
                "plain_copy_s": _seconds(metrics["host_rank_encoded_copy_s"]),
                "de_stream_cuda_s": _seconds(cuda["decode"]) / 1000 if cuda is not None else None,
                "matrix_apply_cuda_s": _seconds(cuda["layout_apply"]) / 1000 if cuda is not None else None,
                "prepare_s": _seconds(metrics["host_prepare_s"]),
                "pause_s": pause,
                "skip_payload_hash": metrics["host_encoded_cache_skip_payload_hash"],
                "receipt": resumed,
            }
        )
    return sorted(rows, key=lambda row: (row["identity"]["engine_id"], row["identity"]["dp_rank"]))


def _owner_compression(metrics, codec):
    enabled, final = metrics["timing_enabled"], metrics["finalization"]
    inner = _seconds(metrics["batch_cuda_s"]["compression_s"]) if enabled else None
    outer = None
    if codec == "lz4":
        outer = 0.0
    elif enabled:
        outer = _seconds(final["finalize_cuda_phase_s"]["outer_zstd_s"])
    return {
        "inner_cuda_s": inner,
        "outer_cuda_s": outer,
        "batch_wall_s": _seconds(metrics["batch_wall_s"]),
        "outer_wall_s": _seconds(final["outer_zstd_wall_s"]),
        "pack_d2h_wall_s": _seconds(final["pack_d2h_wall_s"]),
        "encode_and_target_write_s": _seconds(metrics["encode_and_target_write_s"]),
        "raw_metrics": metrics,
    }


def _compression(row, codec, assignments):
    metrics = row["compression"]["owners"]
    by_owner = {owner["owner"]: owner for owner in metrics}
    if len(by_owner) != len(metrics) or set(by_owner) != set(assignments):
        raise ValueError("Missing or duplicate compression owners")
    for owner, value in by_owner.items():
        if any(value[key] != assignments[owner][key] for key in ("gpu", "groups", "canonical_bytes")):
            raise ValueError("Compression owner assignment differs")
    owners = [_owner_compression(value, codec) for value in metrics]
    if not owners or sum(owner["raw_metrics"]["canonical_bytes"] for owner in owners) != row["canonical_bytes"]:
        raise ValueError("Compression owners do not cover the canonical bytes")
    return {
        "owner_max": {
            key: _maximum(owners, key)
            for key in (
                "inner_cuda_s",
                "outer_cuda_s",
                "batch_wall_s",
                "outer_wall_s",
                "pack_d2h_wall_s",
                "encode_and_target_write_s",
            )
        },
        "seal_s": _seconds(row["seal_s"]),
        "owners": owners,
    }


def _maximum(rows, key):
    values = [row[key] for row in rows]
    return None if any(value is None for value in values) else max(values)


def _format(value):
    return "unmeasured" if value is None else f"{value:.9f}"


def _markdown(summary):
    lines = [
        "# GPU delta end-to-end benchmark",
        "",
        f"Codec: `{summary['codec']}`. Final selected outputs match the independently loaded target checkpoint on "
        f"all {len(summary['comparison']['routes'])} routes, including text, token IDs and input/output logprobs.",
        "",
        f"Sender owners: {summary['sender_count']}. Layer-ordered assignments and all per-owner metrics are retained in `summary.json`.",
        "",
        "Versions are cumulative synthetic targets, not learned updates or statistical repeats. "
        "Selected outputs do not prove every weight byte. Startup and oracle generation are outside update timing.",
        "",
        "Compression columns are independent owner maxima. Inner CUDA is the maximum of each owner's summed batch events; "
        "these maxima can come from different owners and are not a synchronized end-to-end latency. "
        "CUDA events cover their named stream regions, including wrapper/launch gaps, not pure kernel busy time. "
        "Batch wall time includes pinned-input transfers, "
        "XOR and metadata waits; packing/D2H is separate. Owner total also includes checkpoint reads/writes, "
        "payload hashing and shard publication writing. Parent sealing is measured separately after all owners finish. "
        "These overlapping scopes must not be added.",
        "",
        "|Version|Inner CUDA max s|Outer CUDA max s|Batch wall max s|Outer wall max s|Pack/D2H wall max s|Owner total max s|Parent seal s|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    compression_keys = (
        "inner_cuda_s",
        "outer_cuda_s",
        "batch_wall_s",
        "outer_wall_s",
        "pack_d2h_wall_s",
        "encode_and_target_write_s",
    )
    for row in summary["versions"]:
        lines.append(
            f"|{row['version']}|"
            + "|".join(_format(row["compression"]["owner_max"][key]) for key in compression_keys)
            + f"|{_format(row['compression']['seal_s'])}|"
        )
    lines += [
        "",
        "Receiver columns are independent maxima over ranks, not rank sums or additive phases. "
        "Outer CPU/plain-copy wall includes raw copies and job submission/drain; worker sums remain in the raw receipts. "
        "DE stream events include zero-fill and nvCOMP enqueue/host gaps; they are not pure hardware busy time. "
        "Matrix apply includes status checks but excludes raw copies and derived refresh. "
        "Pause comes from completed scheduler pause/resume timestamps. "
        "Disabled CUDA timing is unmeasured; plain LZ4 has no outer stage.",
        "",
        "|Version|Outer CPU max s|Plain copy max s|DE stream CUDA max s|Matrix apply CUDA max s|Prepare max s|Pause max s|Coordinator s|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rank_keys = ("outer_cpu_s", "plain_copy_s", "de_stream_cuda_s", "matrix_apply_cuda_s", "prepare_s", "pause_s")
    for row in summary["versions"]:
        values = [_format(row["receiver_max"][key]) for key in rank_keys] + [_format(row["coordinator_s"])]
        lines.append(f"|{row['version']}|" + "|".join(values) + "|")
    lines += [
        "",
        "|Version|Sender payload checksum|Receiver skips payload SHA|Inner frame bytes|Canonical bytes|Changed bytes|Inner compressed bytes|Matrix payload bytes|Payload file bytes|Manifest bytes|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["versions"]:
        values = [
            row["sender_payload_checksum_format"],
            row["receiver_skip_payload_hash"],
            row["frame_bytes"],
            row["canonical_bytes"],
            row["changed_bytes"],
        ]
        values += [
            row["accounting"][key]
            for key in ("encoded_frame_bytes", "matrix_payload_bytes", "payload_file_bytes", "manifest_bytes")
        ]
        lines.append(f"|{row['version']}|" + "|".join(map(str, values)) + "|")
    lines += [
        "",
        "## Raw owner timings",
        "",
        "|Version|Owner|GPU|Canonical bytes|Inner CUDA s|Outer CUDA s|Batch wall s|Outer wall s|Pack/D2H wall s|Owner total s|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for version in summary["versions"]:
        for owner in version["compression"]["owners"]:
            metrics = owner["raw_metrics"]
            lines.append(
                f"|{version['version']}|{metrics['owner']}|{metrics['gpu']}|{metrics['canonical_bytes']}|"
                + "|".join(_format(owner[key]) for key in compression_keys)
                + "|"
            )
    lines += [
        "",
        "## Raw rank timings",
        "",
        "Full receipts and encoder metrics are retained in `summary.json`.",
        "",
        "|Version|Engine|DP rank|Rank ID|Outer CPU s|Plain copy s|DE stream CUDA s|Matrix apply CUDA s|Prepare s|Pause s|",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for version in summary["versions"]:
        for row in version["ranks"]:
            identity = row["identity"]
            lines.append(
                f"|{version['version']}|{identity['engine_id']}|{identity['dp_rank']}|{identity['rank_id']}|"
                + "|".join(_format(row[key]) for key in rank_keys)
                + "|"
            )
    return "\n".join(lines) + "\n"


def write_report(output: Path):
    fixture = _read(output / "fixture" / "fixture.json")
    codec = fixture["codec"]
    sender_owners = fixture["sender_owners"]
    assignments = {owner["owner"]: owner for owner in sender_owners}
    if len(assignments) != len(sender_owners) or len(assignments) != fixture["sender_gpus"]:
        raise ValueError("Sender owner plan differs from the fixture")
    inventory = _read(output / "receiver" / "inventory.json")
    original = [rank["identity"] for engine in inventory["descriptions"] for rank in engine["participants"]]
    identities = {identity["rank_id"]: identity for identity in original}
    routes = {(identity["engine_id"], identity["dp_rank"]) for identity in original}
    if not identities or len(identities) != len(original) or len(routes) != len(original):
        raise ValueError("Missing or duplicate receiver identities/routes")
    versions = []
    for expected, row in enumerate(fixture["rounds"], 1):
        version = row["version"]
        publication = row["publications"][codec]
        if (
            version != expected
            or publication["base_version"] != version - 1
            or publication["target_version"] != version
        ):
            raise ValueError("Fixture versions are not consecutive")
        update = _read(output / "receiver" / f"update-{version}.json")
        if update["version"] != version or update["codec"] != codec:
            raise ValueError("Update version/codec differs")
        ranks = _rank_rows(update, publication, identities)
        hash_policies = {rank["skip_payload_hash"] for rank in ranks}
        if len(hash_policies) != 1:
            raise ValueError("Receiver ranks have different payload hash policies")
        selected = _generations(_read(output / "receiver" / f"generation-{version}.json"), routes, version)
        versions.append(
            {
                "version": version,
                "frame_bytes": publication["frame_bytes"],
                "sender_payload_checksum_format": publication["payload_checksum_format"],
                "receiver_skip_payload_hash": hash_policies.pop(),
                "canonical_bytes": row["canonical_bytes"],
                "changed_bytes": row["changed_bytes"],
                "accounting": row["accounting"][codec],
                "compression": _compression(row, codec, assignments),
                "coordinator_s": _seconds(update["coordinator_s"]),
                "receiver_max": {
                    key: _maximum(ranks, key)
                    for key in (
                        "outer_cpu_s",
                        "plain_copy_s",
                        "de_stream_cuda_s",
                        "matrix_apply_cuda_s",
                        "prepare_s",
                        "pause_s",
                    )
                },
                "ranks": ranks,
            }
        )
    if not versions:
        raise ValueError("No fixture updates")
    oracle = _generations(_read(output / "oracle" / "target-generation.json"), routes)
    comparison = _compare(selected, oracle)
    _write(output / "comparison.json", comparison)
    if comparison["status"] != "PASS":
        raise ValueError("Final selected generation differs from target checkpoint; see comparison.json")
    summary = {
        "status": "PASS",
        "codec": codec,
        "target_checkpoint": fixture["target_checkpoint"],
        "sender_count": fixture["sender_gpus"],
        "sender_owners": sender_owners,
        "comparison": comparison,
        "versions": versions,
    }
    report = _markdown(summary)
    _write(output / "summary.json", summary)
    (output / "REPORT.md").write_text(report)
    return summary
