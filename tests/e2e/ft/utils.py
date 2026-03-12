"""E2E test utilities for FT system."""

from __future__ import annotations

import logging

from miles.utils.ft.adapters.impl.k8s_node_manager import K8sNodeManager

logger = logging.getLogger(__name__)


async def clear_all_bad_node_markers(node_mgr: K8sNodeManager) -> None:
    """Remove bad-node labels from all currently marked K8s nodes.

    Uses K8sNodeManager internals (_patch_node_labels) since unmark is not
    part of the production NodeManagerProtocol.
    """
    bad_nodes = await node_mgr.get_bad_nodes()
    for node_id in bad_nodes:
        try:
            await node_mgr._patch_node_labels(
                node_id=node_id,
                labels={node_mgr._label_key: None, node_mgr._reason_label_key: None},
            )
        except Exception:
            logger.warning("clear_bad_node_marker_failed node_id=%s", node_id, exc_info=True)
