from typing import Any

import pytest

from miles.utils import object_store_config


class TestParseSize:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (12345, 12345),
            ("12345", 12345),
            ("1kb", 1024),
            ("2k", 2 * 1024),
            ("64mb", 64 * 1024**2),
            ("3m", 3 * 1024**2),
            ("2gb", 2 * 1024**3),
            ("1g", 1024**3),
            ("1.5gb", int(1.5 * 1024**3)),
            ("  2GB ", 2 * 1024**3),
        ],
    )
    def test_parses_ints_and_unit_suffixes(self, value: Any, expected: int):
        """_parse_size handles ints, plain digit strings, and kb/mb/gb suffixes case-insensitively."""
        assert object_store_config._parse_size(value) == expected

    def test_rejects_garbage(self):
        """_parse_size raises ValueError on a non-numeric string without a known unit."""
        with pytest.raises(ValueError):
            object_store_config._parse_size("lots")


class TestMooncakeStoreConfig:
    def _base_kwargs(self) -> dict[str, Any]:
        return {
            "local_hostname": "10.0.0.1",
            "master_server_address": "10.0.0.2:50051",
            "protocol": "tcp",
            "global_segment_size": "2gb",
            "local_buffer_size": "1gb",
        }

    def test_contributing_process_parses_segment_size(self):
        """A contributing process gets the configured global_segment_size parsed to bytes."""
        config = object_store_config.compute_mooncake_store_config(self._base_kwargs(), contribute_segment=True)
        assert config["global_segment_size"] == 2 * 1024**3
        assert config["local_buffer_size"] == 1024**3
        assert config["master_server_addr"] == "10.0.0.2:50051"
        assert config["protocol"] == "tcp"
        assert config["local_hostname"] == "10.0.0.1"

    def test_non_contributing_process_gets_zero_segment(self):
        """A non-contributing process passes global_segment_size=0 (pure client semantics)."""
        config = object_store_config.compute_mooncake_store_config(self._base_kwargs(), contribute_segment=False)
        assert config["global_segment_size"] == 0

    def test_env_fallbacks(self, monkeypatch: pytest.MonkeyPatch):
        """Unset kwargs fall back to MOONCAKE_* environment variables."""
        monkeypatch.setenv("MOONCAKE_LOCAL_HOSTNAME", "10.1.1.1")
        monkeypatch.setenv("MOONCAKE_MASTER", "10.1.1.2:50051")
        monkeypatch.setenv("MOONCAKE_PROTOCOL", "tcp")
        monkeypatch.setenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", "64mb")
        config = object_store_config.compute_mooncake_store_config({}, contribute_segment=True)
        assert config["local_hostname"] == "10.1.1.1"
        assert config["master_server_addr"] == "10.1.1.2:50051"
        assert config["protocol"] == "tcp"
        assert config["global_segment_size"] == 64 * 1024**2

    def test_kwargs_take_precedence_over_env(self, monkeypatch: pytest.MonkeyPatch):
        """Explicit init kwargs win over MOONCAKE_* environment variables."""
        monkeypatch.setenv("MOONCAKE_MASTER", "10.9.9.9:50051")
        config = object_store_config.compute_mooncake_store_config(self._base_kwargs(), contribute_segment=True)
        assert config["master_server_addr"] == "10.0.0.2:50051"

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch):
        """With no kwargs and no env, protocol/metadata/segment sizes use built-in defaults."""
        _clear_mooncake_env(monkeypatch)
        monkeypatch.setattr(object_store_config, "_local_hostname", lambda: "127.0.0.1")
        config = object_store_config.compute_mooncake_store_config({}, contribute_segment=True)
        assert config["protocol"] == "rdma"
        assert config["metadata_server"] == "P2PHANDSHAKE"
        assert config["global_segment_size"] == 8 * 1024**3
        assert config["local_buffer_size"] == 32 * 1024**3
        assert config["master_server_addr"] == ""


class TestMooncakeInitKwargs:
    def test_defaults_name_the_launched_master(self, monkeypatch: pytest.MonkeyPatch):
        """The launcher's defaults point at the master it starts and never read the environment."""
        monkeypatch.setenv("MOONCAKE_MASTER", "10.9.9.9:50051")
        init_kwargs = object_store_config.compute_mooncake_init_kwargs_vanilla(host="10.0.0.2", master_port=1234)
        assert init_kwargs["master_server_address"] == "10.0.0.2:1234"
        assert init_kwargs["protocol"] == "tcp"

    def test_env_answers_only_defaulted_fields(self, monkeypatch: pytest.MonkeyPatch):
        """The environment layer covers every defaulted field and nothing the launcher does not default."""
        _clear_mooncake_env(monkeypatch)
        monkeypatch.setenv("MOONCAKE_MASTER", "10.1.1.2:50051")
        monkeypatch.setenv("MOONCAKE_PROTOCOL", "rdma")
        monkeypatch.setenv("MOONCAKE_LOCAL_HOSTNAME", "10.1.1.1")
        from_env = object_store_config.compute_mooncake_init_kwargs_from_env()
        assert from_env == {"master_server_address": "10.1.1.2:50051", "protocol": "rdma"}

    def test_env_overrides_reach_the_store_config(self, monkeypatch: pytest.MonkeyPatch):
        """A defaulted field the environment names wins over the built-in default at store setup."""
        _clear_mooncake_env(monkeypatch)
        monkeypatch.setenv("MOONCAKE_MASTER", "10.1.1.2:50051")
        monkeypatch.setenv("MOONCAKE_PROTOCOL", "rdma")
        init_kwargs = (
            object_store_config.compute_mooncake_init_kwargs_vanilla()
            | object_store_config.compute_mooncake_init_kwargs_from_env()
        )
        config = object_store_config.compute_mooncake_store_config(init_kwargs, contribute_segment=True)
        assert config["master_server_addr"] == "10.1.1.2:50051"
        assert config["protocol"] == "rdma"


class TestTheEnvironmentAnswersEveryDefaultedField:
    def test_the_capacities_the_platform_names_reach_the_init_kwargs(self, monkeypatch: pytest.MonkeyPatch):
        """A pod sized by its platform would otherwise be given the launcher's 2 GiB of each capacity."""
        _clear_mooncake_env(monkeypatch)
        monkeypatch.setenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", "64mb")
        monkeypatch.setenv("MOONCAKE_LOCAL_BUFFER_SIZE", "16mb")

        from_env = object_store_config.compute_mooncake_init_kwargs_from_env()

        assert from_env == {"global_segment_size": "64mb", "local_buffer_size": "16mb"}

    def test_the_capacities_the_platform_names_survive_into_the_store_config(self, monkeypatch: pytest.MonkeyPatch):
        """The kwargs are only worth answering because this is what the store is finally set up with."""
        _clear_mooncake_env(monkeypatch)
        monkeypatch.setenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", "64mb")

        init_kwargs = (
            object_store_config.compute_mooncake_init_kwargs_vanilla()
            | object_store_config.compute_mooncake_init_kwargs_from_env()
        )
        config = object_store_config.compute_mooncake_store_config(init_kwargs, contribute_segment=True)

        assert config["global_segment_size"] == 64 * 1024**2
        assert config["local_buffer_size"] == 2 * 1024**3

    def test_a_defaulted_field_the_environment_leaves_alone_keeps_the_launcher_default(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The launcher still starts the master it configures, and a run naming nothing must still find it."""
        _clear_mooncake_env(monkeypatch)
        monkeypatch.setenv("MOONCAKE_PROTOCOL", "rdma")

        init_kwargs = (
            object_store_config.compute_mooncake_init_kwargs_vanilla(host="10.0.0.2")
            | object_store_config.compute_mooncake_init_kwargs_from_env()
        )

        assert init_kwargs["master_server_address"] == "10.0.0.2:50051"
        assert init_kwargs["protocol"] == "rdma"

    def test_an_environment_that_names_nothing_changes_nothing(self, monkeypatch: pytest.MonkeyPatch):
        """Every kubernetes run that names no store took this path before the environment was read at all."""
        _clear_mooncake_env(monkeypatch)

        assert object_store_config.compute_mooncake_init_kwargs_from_env() == {}


def _clear_mooncake_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "MOONCAKE_LOCAL_HOSTNAME",
        "MOONCAKE_TE_META_DATA_SERVER",
        "MOONCAKE_LOCAL_BUFFER_SIZE",
        "MOONCAKE_PROTOCOL",
        "MOONCAKE_DEVICE",
        "MOONCAKE_MASTER",
        "MOONCAKE_GLOBAL_SEGMENT_SIZE",
    ):
        monkeypatch.delenv(name, raising=False)


class TestComputeMooncakeInitKwargs:
    def test_custom_host_and_port_produce_the_complete_mooncake_configuration(self) -> None:
        """A custom endpoint produces every required Mooncake initialization setting."""
        assert object_store_config.compute_mooncake_init_kwargs_vanilla(host="store.internal", master_port=60000) == {
            "protocol": "tcp",
            "master_server_address": "store.internal:60000",
            "global_segment_size": "2gb",
            "local_buffer_size": "2gb",
        }
