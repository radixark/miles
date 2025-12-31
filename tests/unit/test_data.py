"""Unit tests for miles/utils/data.py

These tests verify the correctness of data loading functions including
the memory-efficient streaming implementations.
"""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


class TestReadFile:
    """Tests for read_file function and helpers."""

    @pytest.fixture
    def sample_jsonl_file(self, tmp_path):
        """Create a sample JSONL file for testing."""
        data = [
            {"text": "Hello world", "label": "greeting", "id": 0},
            {"text": "How are you?", "label": "question", "id": 1},
            {"text": "Goodbye", "label": "farewell", "id": 2},
            {"text": "Thanks", "label": "gratitude", "id": 3},
            {"text": "Help me", "label": "request", "id": 4},
        ]
        file_path = tmp_path / "test.jsonl"
        with open(file_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        return file_path

    @pytest.fixture
    def sample_parquet_file(self, tmp_path):
        """Create a sample Parquet file for testing."""
        data = {
            "text": ["Hello world", "How are you?", "Goodbye", "Thanks", "Help me"],
            "label": ["greeting", "question", "farewell", "gratitude", "request"],
            "id": [0, 1, 2, 3, 4],
        }
        df = pd.DataFrame(data)
        file_path = tmp_path / "test.parquet"
        # Create with multiple row groups to test chunked reading
        table = pa.Table.from_pandas(df)
        pq.write_table(table, file_path, row_group_size=2)
        return file_path

    @pytest.fixture
    def large_jsonl_file(self, tmp_path):
        """Create a larger JSONL file for testing chunked reading."""
        data = [{"text": f"Item {i}", "label": str(i), "id": i} for i in range(100)]
        file_path = tmp_path / "large.jsonl"
        with open(file_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        return file_path

    def test_read_jsonl_all_rows(self, sample_jsonl_file):
        """Test reading all rows from JSONL file."""
        from miles.utils.data import read_file

        rows = list(read_file(str(sample_jsonl_file)))

        assert len(rows) == 5
        assert rows[0]["text"] == "Hello world"
        assert rows[4]["text"] == "Help me"

    def test_read_parquet_all_rows(self, sample_parquet_file):
        """Test reading all rows from Parquet file."""
        from miles.utils.data import read_file

        rows = list(read_file(str(sample_parquet_file)))

        assert len(rows) == 5
        assert rows[0]["text"] == "Hello world"
        assert rows[4]["text"] == "Help me"

    def test_read_jsonl_with_slice(self, sample_jsonl_file):
        """Test reading JSONL file with slice notation."""
        from miles.utils.data import read_file

        # Read rows 1-3 (indices 1, 2)
        path_with_slice = f"{sample_jsonl_file}@[1:3]"
        rows = list(read_file(path_with_slice))

        assert len(rows) == 2
        assert rows[0]["text"] == "How are you?"
        assert rows[1]["text"] == "Goodbye"

    def test_read_parquet_with_slice(self, sample_parquet_file):
        """Test reading Parquet file with slice notation."""
        from miles.utils.data import read_file

        # Read rows 2-4 (indices 2, 3)
        path_with_slice = f"{sample_parquet_file}@[2:4]"
        rows = list(read_file(path_with_slice))

        assert len(rows) == 2
        assert rows[0]["text"] == "Goodbye"
        assert rows[1]["text"] == "Thanks"

    def test_read_jsonl_slice_from_start(self, sample_jsonl_file):
        """Test reading JSONL file with slice from start."""
        from miles.utils.data import read_file

        path_with_slice = f"{sample_jsonl_file}@[:2]"
        rows = list(read_file(path_with_slice))

        assert len(rows) == 2
        assert rows[0]["text"] == "Hello world"
        assert rows[1]["text"] == "How are you?"

    def test_read_jsonl_slice_to_end(self, sample_jsonl_file):
        """Test reading JSONL file with slice to end."""
        from miles.utils.data import read_file

        path_with_slice = f"{sample_jsonl_file}@[3:]"
        rows = list(read_file(path_with_slice))

        assert len(rows) == 2
        assert rows[0]["text"] == "Thanks"
        assert rows[1]["text"] == "Help me"

    def test_read_file_not_found(self, tmp_path):
        """Test that FileNotFoundError is raised for missing files."""
        from miles.utils.data import read_file

        with pytest.raises(FileNotFoundError):
            list(read_file(str(tmp_path / "nonexistent.jsonl")))

    def test_read_unsupported_format(self, tmp_path):
        """Test that ValueError is raised for unsupported formats."""
        from miles.utils.data import read_file

        # Create a file with unsupported extension
        file_path = tmp_path / "test.csv"
        file_path.write_text("a,b,c\n1,2,3\n")

        with pytest.raises(ValueError, match="Unsupported file format"):
            list(read_file(str(file_path)))

    def test_chunked_reading_jsonl(self, large_jsonl_file):
        """Test that chunked reading works correctly for JSONL."""
        from miles.utils.data import read_file

        # Use small chunk size to force multiple chunks
        rows = list(read_file(str(large_jsonl_file), chunk_size=10))

        assert len(rows) == 100
        assert rows[0]["id"] == 0
        assert rows[99]["id"] == 99

    def test_chunked_reading_with_slice_crossing_chunks(self, large_jsonl_file):
        """Test slicing that crosses chunk boundaries."""
        from miles.utils.data import read_file

        # Slice from row 25 to 35 (10 rows across multiple chunks of size 10)
        path_with_slice = f"{large_jsonl_file}@[25:35]"
        rows = list(read_file(path_with_slice, chunk_size=10))

        assert len(rows) == 10
        assert rows[0]["id"] == 25
        assert rows[9]["id"] == 34

    def test_empty_jsonl_file(self, tmp_path):
        """Test reading empty JSONL file."""
        from miles.utils.data import read_file

        file_path = tmp_path / "empty.jsonl"
        file_path.write_text("")

        rows = list(read_file(str(file_path)))
        assert len(rows) == 0

    def test_label_as_string(self, tmp_path):
        """Test that labels are read as strings (not converted to int/float)."""
        from miles.utils.data import read_file

        data = [{"text": "test", "label": "123"}]
        file_path = tmp_path / "test.jsonl"
        with open(file_path, "w") as f:
            f.write(json.dumps(data[0]) + "\n")

        rows = list(read_file(str(file_path)))
        assert isinstance(rows[0]["label"], str)
        assert rows[0]["label"] == "123"


class TestParseGeneralizedPath:
    """Tests for _parse_generalized_path function."""

    def test_simple_path(self):
        """Test parsing a simple path without slice."""
        from miles.utils.data import _parse_generalized_path

        path, row_slice = _parse_generalized_path("/path/to/file.jsonl")

        assert path == "/path/to/file.jsonl"
        assert row_slice is None

    def test_path_with_full_slice(self):
        """Test parsing path with start and end slice."""
        from miles.utils.data import _parse_generalized_path

        path, row_slice = _parse_generalized_path("/path/to/file.jsonl@[10:20]")

        assert path == "/path/to/file.jsonl"
        assert row_slice == slice(10, 20)

    def test_path_with_start_only_slice(self):
        """Test parsing path with start-only slice."""
        from miles.utils.data import _parse_generalized_path

        path, row_slice = _parse_generalized_path("/path/to/file.jsonl@[5:]")

        assert path == "/path/to/file.jsonl"
        assert row_slice == slice(5, None)

    def test_path_with_end_only_slice(self):
        """Test parsing path with end-only slice."""
        from miles.utils.data import _parse_generalized_path

        path, row_slice = _parse_generalized_path("/path/to/file.jsonl@[:15]")

        assert path == "/path/to/file.jsonl"
        assert row_slice == slice(None, 15)

    def test_path_with_negative_indices(self):
        """Test parsing path with negative slice indices."""
        from miles.utils.data import _parse_generalized_path

        path, row_slice = _parse_generalized_path("/path/to/file.jsonl@[-10:-5]")

        assert path == "/path/to/file.jsonl"
        assert row_slice == slice(-10, -5)


class TestGetMinimumNumMicroBatchSize:
    """Tests for get_minimum_num_micro_batch_size function."""

    def test_single_batch(self):
        """Test when all sequences fit in one batch."""
        from miles.utils.data import get_minimum_num_micro_batch_size

        total_lengths = [100, 200, 300]
        max_tokens = 1000

        num_batches = get_minimum_num_micro_batch_size(total_lengths, max_tokens)

        assert num_batches == 1

    def test_multiple_batches(self):
        """Test when sequences need multiple batches."""
        from miles.utils.data import get_minimum_num_micro_batch_size

        total_lengths = [400, 400, 400]
        max_tokens = 500

        num_batches = get_minimum_num_micro_batch_size(total_lengths, max_tokens)

        assert num_batches == 3  # Each needs its own batch

    def test_first_fit_packing(self):
        """Test first-fit bin packing behavior."""
        from miles.utils.data import get_minimum_num_micro_batch_size

        # Two small items can pack together, large one needs its own batch
        total_lengths = [200, 800, 200]
        max_tokens = 500

        num_batches = get_minimum_num_micro_batch_size(total_lengths, max_tokens)

        # 200 + 200 = 400 fits in one batch, 800 needs its own
        assert num_batches == 2

    def test_empty_input(self):
        """Test with empty input."""
        from miles.utils.data import get_minimum_num_micro_batch_size

        num_batches = get_minimum_num_micro_batch_size([], 1000)

        assert num_batches == 0
