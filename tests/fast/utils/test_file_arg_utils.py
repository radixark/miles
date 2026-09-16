import base64
import binascii

import pytest

from miles.utils.file_arg_utils import PSEUDO_FILE_PREFIX, resolve_file_arg


class TestResolveFileArg:
    def test_reads_a_plain_file_path(self, tmp_path):
        """A bare path keeps working, so existing launchers are unaffected."""
        path = tmp_path / "config.yaml"
        path.write_text("a: 1\n")

        assert resolve_file_arg(str(path)) == "a: 1\n"

    def test_decodes_an_inline_base64_payload(self):
        """An inline payload needs no shared filesystem between launcher and worker."""
        encoded = base64.b64encode(b"a: 1\n").decode()

        assert resolve_file_arg(f"{PSEUDO_FILE_PREFIX}{encoded}") == "a: 1\n"

    def test_round_trips_multiline_utf8_content(self):
        """Config documents are multi-line and may carry non-ascii comments."""
        text = "eval:\n  datasets:\n    - name: aime  # 中文注释\n"
        encoded = base64.b64encode(text.encode()).decode()

        assert resolve_file_arg(f"{PSEUDO_FILE_PREFIX}{encoded}") == text

    def test_reads_a_utf8_file_whatever_the_process_locale_is(self, tmp_path):
        """OmegaConf.load() always read UTF-8, so a non-UTF-8 default locale must not change the result."""
        path = tmp_path / "config.yaml"
        text = "eval:\n  name: aime  # 中文注释\n"
        path.write_text(text, encoding="utf-8")

        assert resolve_file_arg(str(path)) == text

    def test_a_missing_path_still_raises(self, tmp_path):
        """A typo in a path must fail loudly rather than silently yield an empty config."""
        with pytest.raises(FileNotFoundError):
            resolve_file_arg(str(tmp_path / "absent.yaml"))

    @pytest.mark.parametrize("payload", ["!!!!", "a: 1", "eval:"], ids=["symbols", "yaml", "truncated"])
    def test_a_corrupt_payload_raises_instead_of_decoding_to_nothing(self, payload):
        """b64decode() drops invalid characters by default, which would silently yield an empty config."""
        with pytest.raises(binascii.Error):
            resolve_file_arg(f"{PSEUDO_FILE_PREFIX}{payload}")
