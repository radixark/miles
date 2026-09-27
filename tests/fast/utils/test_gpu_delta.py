from unittest.mock import MagicMock

import pytest

from miles.utils.gpu_delta import PendingBatch


def test_failed_payload_drain_retains_storage_until_retry_succeeds():
    stream, ready = MagicMock(), MagicMock()
    encoded, keepalive = (object(),), (object(),)
    pending = PendingBatch(encoded=encoded, metadata=None, totals=(), stream=stream, ready=ready, keepalive=keepalive)
    # A failed payload-copy event leaves compression complete but its copy
    # stream still requiring a drain before source buffers may be released.
    pending._copy_pending = True
    stream.synchronize.side_effect = [OSError("copy drain failed"), None]

    with pytest.raises(OSError, match="copy drain failed"):
        pending.close()
    assert pending._encoded is encoded and pending._keepalive is keepalive
    assert pending._copy_pending and not pending._closed

    pending.close()
    pending.close()
    assert pending._closed and not pending._copy_pending
    assert pending._encoded == pending._keepalive == ()
    assert stream.synchronize.call_count == 2
