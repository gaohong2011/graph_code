"""Unit tests for file_tools module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from graph_code.tools.file_tools import read_file, write_file, list_directory, glob_search
from graph_code.config import get_config, reset_config


class TestReadFile:
    """Tests for read_file function."""

    def test_read_existing_file(self, tmp_path):
        """Test reading an existing file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("line1\nline2\nline3\n")

        with patch('graph_code.tools.file_tools.get_config') as mock_get_config:
            mock_config = type('Config', (), {'working_path': tmp_path})()
            mock_get_config.return_value = mock_config

            result = read_file("test.txt")

        assert "line1" in result
        assert "line2" in result
        assert "line3" in result
        assert "File: test.txt" in result

    def test_read_file_with_offset_and_limit(self, tmp_path):
        """Test reading file with offset and limit."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("line1\nline2\nline3\nline4\nline5\n")

        with patch('graph_code.tools.file_tools.get_config') as mock_get_config:
            mock_config = type('Config', (), {'working_path': tmp_path})()
            mock_get_config.return_value = mock_config

            result = read_file("test.txt", offset=2, limit=2)

        assert "line3" in result
        assert "line4" in result
        assert "line1" not in result
        assert "line5" not in result

    def test_read_nonexistent_file(self, tmp_path):
        """Test reading a non-existent file returns error."""
        with patch('graph_code.tools.file_tools.get_config') as mock_get_config:
            mock_config = type('Config', (), {'working_path': tmp_path})()
            mock_get_config.return_value = mock_config

            result = read_file("nonexistent.txt")

        assert "Error: File not found" in result

    def test_read_file_outside_working_dir(self, tmp_path):
        """Test that reading file outside working directory is blocked."""
        with patch('graph_code.tools.file_tools.get_config') as mock_get_config:
            mock_config = type('Config', (), {'working_path': tmp_path})()
            mock_get_config.return_value = mock_config

            result = read_file("/etc/passwd")

        assert "Access denied" in result


class TestWriteFile:
    """Tests for write_file function."""

    def test_write_new_file(self, tmp_path):
        """Test writing to a new file."""
        with patch('graph_code.tools.file_tools.get_config') as mock_get_config:
            mock_config = type('Config', (), {'working_path': tmp_path})()
            mock_get_config.return_value = mock_config

            result = write_file("newfile.txt", "Hello, World!")

        assert "Wrote file" in result
        assert (tmp_path / "newfile.txt").read_text() == "Hello, World!"

    def test_append_to_file(self, tmp_path):
        """Test appending to an existing file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Original content\n")

        with patch('graph_code.tools.file_tools.get_config') as mock_get_config:
            mock_config = type('Config', (), {'working_path': tmp_path})()
            mock_get_config.return_value = mock_config

            result = write_file("test.txt", "Appended content", append=True)

        assert "Appended to" in result
        content = test_file.read_text()
        assert "Original content" in content
        assert "Appended content" in content

    def test_create_nested_directory(self, tmp_path):
        """Test that write_file creates parent directories."""
        with patch('graph_code.tools.file_tools.get_config') as mock_get_config:
            mock_config = type('Config', (), {'working_path': tmp_path})()
            mock_get_config.return_value = mock_config

            result = write_file("nested/dir/file.txt", "content")

        assert "Wrote file" in result
        assert (tmp_path / "nested" / "dir" / "file.txt").exists()

    def test_write_file_outside_working_dir(self, tmp_path):
        """Test that writing file outside working directory is blocked."""
        with patch('graph_code.tools.file_tools.get_config') as mock_get_config:
            mock_config = type('Config', (), {'working_path': tmp_path})()
            mock_get_config.return_value = mock_config

            result = write_file("/tmp/outside.txt", "content")

        assert "Access denied" in result


class TestListDirectory:
    """Tests for list_directory function."""

    def test_list_directory_contents(self, tmp_path):
        """Test listing directory contents."""
        (tmp_path / "file1.txt").write_text("content")
        (tmp_path / "file2.py").write_text("content")
        (tmp_path / "subdir").mkdir()

        with patch('graph_code.tools.file_tools.get_config') as mock_get_config:
            mock_config = type('Config', (), {'working_path': tmp_path})()
            mock_get_config.return_value = mock_config

            result = list_directory(".")

        assert "file1.txt" in result
        assert "file2.py" in result
        assert "subdir" in result

    def test_list_nonexistent_directory(self, tmp_path):
        """Test listing non-existent directory returns error."""
        with patch('graph_code.tools.file_tools.get_config') as mock_get_config:
            mock_config = type('Config', (), {'working_path': tmp_path})()
            mock_get_config.return_value = mock_config

            result = list_directory("nonexistent")

        assert "Error: Directory not found" in result

    def test_list_directory_recursive(self, tmp_path):
        """Test recursive directory listing."""
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "nested.txt").write_text("content")

        with patch('graph_code.tools.file_tools.get_config') as mock_get_config:
            mock_config = type('Config', (), {'working_path': tmp_path})()
            mock_get_config.return_value = mock_config

            result = list_directory(".", recursive=True)

        assert "subdir" in result
        assert "nested.txt" in result


class TestGlobSearch:
    """Tests for glob_search function."""

    def test_glob_search_pattern(self, tmp_path):
        """Test searching files with glob pattern."""
        (tmp_path / "test1.py").write_text("content")
        (tmp_path / "test2.py").write_text("content")
        (tmp_path / "readme.txt").write_text("content")

        with patch('graph_code.tools.file_tools.get_config') as mock_get_config:
            mock_config = type('Config', (), {'working_path': tmp_path})()
            mock_get_config.return_value = mock_config

            result = glob_search("*.py")

        assert "test1.py" in result
        assert "test2.py" in result
        assert "readme.txt" not in result

    def test_glob_search_recursive(self, tmp_path):
        """Test recursive glob search."""
        (tmp_path / "root.py").write_text("content")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.py").write_text("content")

        with patch('graph_code.tools.file_tools.get_config') as mock_get_config:
            mock_config = type('Config', (), {'working_path': tmp_path})()
            mock_get_config.return_value = mock_config

            result = glob_search("**/*.py")

        assert "root.py" in result
        assert "nested.py" in result

    def test_glob_search_no_matches(self, tmp_path):
        """Test glob search with no matches."""
        with patch('graph_code.tools.file_tools.get_config') as mock_get_config:
            mock_config = type('Config', (), {'working_path': tmp_path})()
            mock_get_config.return_value = mock_config

            result = glob_search("*.nonexistent")

        assert "No files found" in result
