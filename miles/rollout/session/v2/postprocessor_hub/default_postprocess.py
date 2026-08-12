from miles.utils.types import Sample

_SERVER_OWNED_METADATA_KEYS = ("accumulated_token_ids", "tito_session_mismatch", "leaf")


def check_input_metadata(agent_metadata: dict) -> dict:
    return {key: value for key, value in agent_metadata.items() if key not in _SERVER_OWNED_METADATA_KEYS}


def assign_reward(samples: list[Sample], trajectory_reward: float) -> None:
    for sample in samples:
        sample.reward = trajectory_reward


def default_postprocess(leaf_samples: list[Sample], session_metadata: dict) -> list[Sample]:
    """Finalize the leaf samples kept by the picker.

    A shared completion is trainable only in the earliest committed kept leaf.
    Ownership is computed after picking, so a dropped leaf cannot own it.

    Agent metadata cannot replace server-owned fields. A provided trajectory reward is assigned to every kept sample.

    Mutates each `Sample` in place and preserves picker order.
    """
    nodes_by_id = {node["id"]: node for node in session_metadata["tree"]["nodes"]}
    agent_metadata = session_metadata.get("agent") or {}
    trajectory_reward = agent_metadata.get("reward")
    agent_sample_metadata = check_input_metadata(agent_metadata)

    # Picker order may change; `node_id` preserves commit order.
    owner_by_node_id: dict[int, int] = {}
    for sample in leaf_samples:
        leaf = sample.metadata["leaf"]
        leaf_id = leaf["node_id"]
        for node_id in leaf["path_node_ids"]:
            owner_by_node_id[node_id] = min(leaf_id, owner_by_node_id.get(node_id, leaf_id))

    processed_samples: list[Sample] = []
    for sample in leaf_samples:
        leaf = sample.metadata["leaf"]
        response_start = len(sample.tokens) - sample.response_length
        # Spans index all tokens; `loss_mask` indexes only the response.
        # Clamp when early-stop or truncation shortens the sample.
        for node_id in leaf["path_node_ids"]:
            if owner_by_node_id[node_id] == leaf["node_id"]:
                continue
            completion_start, completion_end = nodes_by_id[node_id]["completion_span"]
            mask_start = max(completion_start - response_start, 0)
            mask_end = min(completion_end - response_start, sample.response_length)
            if mask_end > mask_start:
                sample.loss_mask[mask_start:mask_end] = [0] * (mask_end - mask_start)
        server_metadata = {key: sample.metadata[key] for key in _SERVER_OWNED_METADATA_KEYS if key in sample.metadata}
        sample.metadata = {**sample.metadata, **agent_sample_metadata, **server_metadata}
        processed_samples.append(sample)
    # Skip `assign_reward` to score downstream, or replace the scalar
    # `agent_metadata["reward"]` for finer-grained assignment.
    if trajectory_reward is not None:
        assign_reward(processed_samples, trajectory_reward)
    return processed_samples
