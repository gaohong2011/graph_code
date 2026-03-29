"""Unit tests for code_tools module."""

from unittest.mock import patch

import pytest

from graph_code.tools.code_tools import grep_search, read_code_chunk


class TestGrepSearch:
    """Tests for grep_search function."""

    def test_grep_search_simple_pattern(self, tmp_path):
        """Test searching with simple pattern."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    pass\n")

        with patch('graph_code.tools.code_tools.get_config') as mock_get_config:
            mock_config = type('Config', (), {'working_path': tmp_path})()
            mock_get_config.return_value = mock_config

            result = grep_search("def hello")

        assert "Found" in result
        assert "test.py" in result
        assert "def hello" in result

    def test_grep_search_regex_pattern(self, tmp_path):
        """Test searching with regex pattern."""
        test_file = tmp_path / "test.py"
        test_file.write_text("variable1 = 1\nvariable2 = 2\n")

        with patch('graph_code.tools.code_tools.get_config') as mock_get_config:
            mock_config = type('Config', (), {'working_path': tmp_path})()
            mock_get_config.return_value = mock_config

            result = grep_search(r"variable\d+")

        assert "variable1" in result
        assert "variable2" in result

    def test_grep_search_invalid_regex(self, tmp_path):
        """Test searching with invalid regex returns error."""
        with patch('graph_code.tools.code_tools.get_config') as mock_get_config:
            mock_config = type('Config', (), {'working_path': tmp_path})()
            mock_get_config.return_value = mock_config

            result = grep_search("[invalid")

        assert "Error: Invalid regex pattern" in result

    def test_grep_search_with_glob_filter(self, tmp_path):
        """Test searching with file glob filter."""
        (tmp_path / "test.py").write_text("search_term = 1\n")
        (tmp_path / "test.txt").write_text("search_term = 2\n")

        with patch('graph_code.tools.code_tools.get_config') as mock_get_config:
            mock_config = type('Config', (), {'working_path': tmp_path})()
            mock_get_config.return_value = mock_config

            result = grep_search("search_term", glob="*.py")

        # Should only match .py file
        matches = result.count("search_term")
        assert matches == 1

    def test_grep_search_no_matches(self, tmp_path):
        """Test searching with no matches."""
        test_file = tmp_path / "test.py"
        test_file.write_text("some code here\n")

        with patch('graph_code.tools.code_tools.get_config') as mock_get_config:
            mock_config = type('Config', (), {'working_path': tmp_path})()
            mock_get_config.return_value = mock_config

            result = grep_search("nonexistent_pattern_12345")

        assert "No matches found" in result

    def test_grep_search_single_file(self, tmp_path):
        """Test searching in a specific file."""
        test_file = tmp_path / "target.py"
        test_file.write_text("find_me = 1\n")
        (tmp_path / "other.py").write_text("find_me = 2\n")

        with patch('graph_code.tools.code_tools.get_config') as mock_get_config:
            mock_config = type('Config', (), {'working_path': tmp_path})()
            mock_get_config.return_value = mock_config

            result = grep_search("find_me", path="target.py")

        assert "target.py" in result

    def test_grep_search_outside_working_dir(self, tmp_path):
        """Test that searching outside working directory is blocked."""
        with patch('graph_code.tools.code_tools.get_config') as mock_get_config:
            mock_config = type('Config', (), {'working_path': tmp_path})()
            mock_get_config.return_value = mock_config

            result = grep_search("pattern", path="/etc")

        assert "Access denied" in result


class TestReadCodeChunk:
    """Tests for read_code_chunk function."""

    def test_read_code_chunk_basic(self, tmp_path):
        """Test reading a basic code chunk."""
        test_file = tmp_path / "test.py"
        test_file.write_text("line1\nline2\nline3\nline4\nline5\n")

        with patch('graph_code.tools.code_tools.get_config') as mock_get_config:
            mock_config = type('Config', (), {'working_path': tmp_path})()
            mock_get_config.return_value = mock_config

            result = read_code_chunk("test.py", start_line=2, end_line=4)

        assert ">>>" in result  # Lines in range are marked
        assert "line2" in result
        assert "line3" in result
        assert "line4" in result

    def test_read_code_chunk_with_context(self, tmp_path):
        """Test reading code chunk with context lines."""
        test_file = tmp_path / "test.py"
        lines = [f"line{i}\n" for i in range(1, 11)]
        test_file.write_text("".join(lines))

        with patch('graph_code.tools.code_tools.get_config') as mock_get_config:
            mock_config = type('Config', (), {'working_path': tmp_path})()
            mock_get_config.return_value = mock_config

            result = read_code_chunk("test.py", start_line=5, end_line=5, context_lines=2)

        # Should show lines 3-7 (5 with 2 context lines)
        assert "line3" in result
        assert "line5" in result
        assert "line7" in result
        # Line 5 should be marked as in range
        assert ">>>    5 |" in result
        # Context lines should not be marked
        assert "       3 |" in result
        assert "       7 |" in result

    def test_read_code_chunk_nonexistent_file(self, tmp_path):
        """Test reading chunk from non-existent file returns error."""
        with patch('graph_code.tools.code_tools.get_config') as mock_get_config:
            mock_config = type('Config', (), {'working_path': tmp_path})()
            mock_get_config.return_value = mock_config

            result = read_code_chunk("nonexistent.py", start_line=1)

        assert "Error: File not found" in result

    def test_read_code_chunk_single_line(self, tmp_path):
        """Test reading a single line (default end_line=start_line)."""
        test_file = tmp_path / "test.py"
        test_file.write_text("line1\nline2\nline3\n")

        with patch('graph_code.tools.code_tools.get_config') as mock_get_config:
            mock_config = type('Config', (), {'working_path': tmp_path})()
            mock_get_config.return_value = mock_config

            result = read_code_chunk("test.py", start_line=2)

        # Only line 2 should be marked
        assert ">>>    2 |" in result
        assert "line2" in result

    def test_read_code_chunk_outside_working_dir(self, tmp_path):
        """Test that reading chunk outside working directory is blocked."""
        with patch('graph_code.tools.code_tools.get_config') as mock_get_config:
            mock_config = type('Config', (), {'working_path': tmp_path})()
            mock_get_config.return_value = mock_config

            result = read_code_chunk("/etc/passwd", start_line=1)

        assert "Access denied" in result
