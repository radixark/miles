from types import SimpleNamespace

import numpy as np
import pybase64
import pytest

from miles.rollout.generate_utils.generate_endpoint_utils import get_indexer_topk_from_response
from miles.utils.types import Sample


class TestIndexerStreamContract:
    @pytest.mark.parametrize("expected_streams", [1, 3])
    def test_model_stream_count_mismatch_is_rejected(self, expected_streams: int) -> None:
        """A server capture cannot be replayed onto a model with different indexer layers."""
        output = {
            "meta_info": {
                "indexer_topk": pybase64.b64encode(np.arange(12, dtype=np.int32).tobytes()).decode("ascii"),
                "indexer_topk_num_layers": 2,
            }
        }

        with pytest.raises(AssertionError, match=f"model has {expected_streams} indexer layers"):
            get_indexer_topk_from_response(
                SimpleNamespace(rollout_indexer_topk_num_streams=expected_streams), output, Sample(tokens=[1, 2, 3])
            )

    def test_matching_model_stream_count_decodes_token_stream_and_topk_axes(self) -> None:
        """Configured stream validation preserves the capture's token and layer ordering."""
        values = np.arange(12, dtype=np.int32)
        output = {
            "meta_info": {
                "indexer_topk": pybase64.b64encode(values.tobytes()).decode("ascii"),
                "indexer_topk_num_layers": 2,
            }
        }

        decoded = get_indexer_topk_from_response(
            SimpleNamespace(rollout_indexer_topk_num_streams=2), output, Sample(tokens=[1, 2, 3])
        )

        np.testing.assert_array_equal(decoded, values.reshape(2, 2, 3))
