from argparse import Namespace
from pathlib import Path
import pytest
from miles.utils.audit_utils.config_snapshot.dumper import ConfigSnapshotDumper
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.test_utils.snapshot import SNAPSHOT_RECORD_DIR_ENV_VAR


@pytest.fixture
def capture_args(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Namespace:
    monkeypatch.setenv(SNAPSHOT_RECORD_DIR_ENV_VAR, str(tmp_path))
    monkeypatch.setattr(ConfigSnapshotDumper, '_state', None)
    args = Namespace(ci_disable_config_snapshot=False, ci_test=True, config_snapshot_name='test/run-0000', deploy_component='all', deploy_instance_id=None, run_uuid='uuid-0', sglang=Namespace(models=[Namespace(name='actor')]), sglang_router_ip=None, sglang_router_port=None, sglang_model_routers=None, use_session_server=True, hf_checkpoint='/model', session_server_workers=2, session_server_ip=None, session_server_port=None, session_server_external_host=None, session_server_instances=None)
    ConfigSnapshotDumper.configure(args=args, source=SimpleProcessIdentity(component='main'))
    ConfigSnapshotDumper.dump(stage='process_config', config={'args': args})
    return args
