from miles.rollout.session.v2.postprocessor_hub.default_postprocess import default_postprocess
from miles.utils.types import Sample


def non_positive_attempts(leaf_samples: list[Sample], session_metadata: dict) -> list[Sample]:
    """Keep failed-attempt context without assigning it positive policy credit.

    The agent must supply ``non_positive_advantage_response_ids`` (possibly
    empty). Match those IDs to the server's completion spans, never to rendered
    text. Dropped branches need no training constraint. Span offsets in sample
    metadata are response-relative and are applied after advantage normalization
    by the trainer; loss masks and trajectory rewards remain unchanged.
    """
    response_ids = (session_metadata.get("agent") or {}).get("non_positive_advantage_response_ids")
    if not isinstance(response_ids, list) or any(not isinstance(rid, str) or not rid for rid in response_ids):
        raise ValueError("Agent must provide non_positive_advantage_response_ids as a list of nonempty strings")

    requested_ids = set(response_ids)
    nodes = session_metadata["tree"]["nodes"]
    matched_nodes = [node for node in nodes if node.get("response_id") in requested_ids]
    matched_ids = [node["response_id"] for node in matched_nodes]
    if len(matched_ids) != len(set(matched_ids)):
        raise ValueError("Non-positive-advantage response IDs must identify unique session nodes")
    if missing := requested_ids - set(matched_ids):
        raise ValueError(f"Non-positive-advantage response IDs are absent from the session tree: {sorted(missing)}")

    invalid_nodes = {node["id"]: node for node in matched_nodes}
    samples = default_postprocess(leaf_samples, session_metadata)
    for sample in samples:
        response_start = len(sample.tokens) - sample.response_length
        spans = []
        for node_id in sample.metadata["leaf"]["path_node_ids"]:
            if node_id not in invalid_nodes:
                continue
            start, end = invalid_nodes[node_id]["completion_span"]
            start = max(start - response_start, 0)
            end = min(end - response_start, sample.response_length)
            if start < end:
                spans.append([start, end])
        # Agent metadata cannot override server-derived token coordinates.
        sample.metadata["non_positive_advantage_spans"] = spans
    return samples
