"""Unit tests for exec_tools module."""

from unittest.mock import patch, MagicMock
import subprocess

import pytest

from graph_code.tools.exec_tools import bash_command, python_execute


class TestBashCommand:
    """Tests for bash_command function."""

    def test_bash_command_success(self, tmp_path):
        """Test successful bash command execution."""
        with patch('graph_code.tools.exec_tools.get_config') as mock_get_config:
            mock_config = type('Config', (), {'working_path': tmp_path})()
            mock_get_config.return_value = mock_config

            result = bash_command("echo 'Hello, World!'")

        assert "Command: echo 'Hello, World!'" in result
        assert "Exit code: 0" in result
        assert "Hello, World!" in result

    def test_bash_command_with_error(self, tmp_path):
        """Test bash command that returns error exit code."""
        with patch('graph_code.tools.exec_tools.get_config') as mock_get_config:
            mock_config = type('Config', (), {'working_path': tmp_path})()
            mock_get_config.return_value = mock_config

            result = bash_command("ls /nonexistent_directory_12345")

        assert "Exit code:" in result
        assert "STDERR:" in result

    def test_bash_command_dangerous_blocked(self, tmp_path):
        """Test that dangerous commands are blocked."""
        with patch('graph_code.tools.exec_tools.get_config') as mock_get_config:
            mock_config = type('Config', (), {'working_path': tmp_path})()
            mock_get_config.return_value = mock_config

            result = bash_command("rm -rf /")

        assert "Error: Potentially dangerous command blocked" in result

    def test_bash_command_timeout(self, tmp_path):
        """Test command timeout."""
        with patch('graph_code.tools.exec_tools.get_config') as mock_get_config:
            mock_config = type('Config', (), {'working_path': tmp_path})()
            mock_get_config.return_value = mock_config

            result = bash_command("sleep 5", timeout=1)

        assert "Error: Command timed out" in result

    def test_bash_command_cwd_set(self, tmp_path):
        """Test that command runs in correct working directory."""
        with patch('graph_code.tools.exec_tools.get_config') as mock_get_config:
            mock_config = type('Config', (), {'working_path': tmp_path})()
            mock_get_config.return_value = mock_config

            result = bash_command("pwd")

        assert str(tmp_path) in result


class TestPythonExecute:
    """Tests for python_execute function."""

    def test_python_execute_success(self, tmp_path):
        """Test successful Python code execution."""
        with patch('graph_code.tools.exec_tools.get_config') as mock_get_config:
            mock_config = type('Config', (), {'working_path': tmp_path})()
            mock_get_config.return_value = mock_config

            result = python_execute("print('Hello from Python')")

        assert "Hello from Python" in result
        assert "Exit code: 0" in result

    def test_python_execute_with_variables(self, tmp_path):
        """Test Python execution with variable manipulation."""
        with patch('graph_code.tools.exec_tools.get_config') as mock_get_config:
            mock_config = type('Config', (), {'working_path': tmp_path})()
            mock_get_config.return_value = mock_config

            code = """
x = 10
y = 20
print(f"Sum: {x + y}")
"""
            result = python_execute(code)

        assert "Sum: 30" in result

    def test_python_execute_syntax_error(self, tmp_path):
        """Test Python execution with syntax error."""
        with patch('graph_code.tools.exec_tools.get_config') as mock_get_config:
            mock_config = type('Config', (), {'working_path': tmp_path})()
            mock_get_config.return_value = mock_config

            result = python_execute("if True print('syntax error')")

        assert "STDERR:" in result
        assert "SyntaxError" in result

    def test_python_execute_runtime_error(self, tmp_path):
        """Test Python execution with runtime error."""
        with patch('graph_code.tools.exec_tools.get_config') as mock_get_config:
            mock_config = type('Config', (), {'working_path': tmp_path})()
            mock_get_config.return_value = mock_config

            result = python_execute("1/0")

        assert "STDERR:" in result
        assert "ZeroDivisionError" in result

    def test_python_execute_timeout(self, tmp_path):
        """Test Python execution timeout."""
        with patch('graph_code.tools.exec_tools.get_config') as mock_get_config:
            mock_config = type('Config', (), {'working_path': tmp_path})()
            mock_get_config.return_value = mock_config

            code = """
import time
time.sleep(10)
"""
            result = python_execute(code, timeout=1)

        assert "Error: Code execution timed out" in result

    def test_python_execute_creates_temp_file(self, tmp_path):
        """Test that Python execution creates and cleans up temp file."""
        with patch('graph_code.tools.exec_tools.get_config') as mock_get_config:
            mock_config = type('Config', (), {'working_path': tmp_path})()
            mock_get_config.return_value = mock_config

            python_execute("print('test')")

            # Check that temp file was cleaned up (no .py files should remain)
            py_files = list(tmp_path.glob("*.py"))
            assert len(py_files) == 0
