from collections.abc import Callable
import pytest
from miles.utils.audit_utils.config_snapshot.converter import ConfigSnapshotConverter


class TestCaptureIntegrity:

    def test_missing_initial_configuration_is_rejected(self, make_record: Callable) -> None:
        """Later stages cannot invent the process baseline needed to interpret their deltas."""
        with pytest.raises(ValueError, match='Missing process_config-0000'):
            ConfigSnapshotConverter.convert([make_record(stage='train_first_step')])

    def test_duplicate_stage_indices_are_rejected(self, make_record: Callable) -> None:
        """Two records claiming the same stage occurrence cannot silently overwrite one another."""
        record = make_record()
        with pytest.raises(ValueError, match='Duplicate snapshot stage'):
            ConfigSnapshotConverter.convert([record, record])

    def test_one_capture_id_cannot_mix_process_contexts(self, make_record: Callable) -> None:
        """Capture IDs cannot merge records from different deployment identities."""
        first = make_record()
        second = make_record(stage='train_first_step')
        second = second.model_copy(update={'context': second.context.model_copy(update={'deploy_instance_id': 'other'})})
        with pytest.raises(ValueError, match='Mixed snapshot contexts'):
            ConfigSnapshotConverter.convert([first, second])
