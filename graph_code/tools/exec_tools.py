"""Execution tools for Graph Code."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from ..config import get_config


def _get_safe_path(file_path: str) -> Path:
    """Get absolute path and ensure it's within working directory."""
    config = get_config()
    working_dir = config.working_path

    if os.path.isabs(file_path):
        target = Path(file_path).resolve()
    else:
        target = (working_dir / file_path).resolve()

    try:
        target.relative_to(working_dir)
    except ValueError:
        raise ValueError(f"Access denied: {file_path} is outside working directory")

    return target


def bash_command(command: str, timeout: int = 60) -> str:
    """Execute a bash command.

    Args:
        command: Bash command to execute
        timeout: Maximum execution time in seconds

    Returns:
        Command output or error message
    """
    config = get_config()
    working_dir = str(config.working_path)

    # Security: Block some dangerous commands
    dangerous_commands = ['rm -rf /', 'rm -rf /*', 'dd if=', ':(){ :|:& };:']
    for dangerous in dangerous_commands:
        if dangerous in command:
            return f"Error: Potentially dangerous command blocked: {dangerous}"

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        output_lines = []
        output_lines.append(f"Command: {command}")
        output_lines.append(f"Exit code: {result.returncode}")
        output_lines.append("-" * 60)

        if result.stdout:
            output_lines.append("STDOUT:")
            output_lines.append(result.stdout)

        if result.stderr:
            output_lines.append("STDERR:")
            output_lines.append(result.stderr)

        return "\n".join(output_lines)

    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout} seconds"
    except Exception as e:
        return f"Error executing command: {e}"


def python_execute(code: str, timeout: int = 30) -> str:
    """Execute Python code.

    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds

    Returns:
        Execution output or error message
    """
    config = get_config()
    working_dir = str(config.working_path)

    # Create a temporary file for the code
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.py',
        dir=working_dir,
        delete=False
    ) as f:
        f.write(code)
        temp_file = f.name

    try:
        # Execute with restricted environment
        env = os.environ.copy()
        env['PYTHONPATH'] = working_dir

        result = subprocess.run(
            [sys.executable, temp_file],
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )

        output_lines = []
        output_lines.append("Python execution result:")
        output_lines.append("-" * 60)

        if result.stdout:
            output_lines.append(result.stdout)

        if result.stderr:
            output_lines.append("STDERR:")
            output_lines.append(result.stderr)

        output_lines.append("-" * 60)
        output_lines.append(f"Exit code: {result.returncode}")

        return "\n".join(output_lines)

    except subprocess.TimeoutExpired:
        return f"Error: Code execution timed out after {timeout} seconds"
    except Exception as e:
        return f"Error executing Python code: {e}"
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_file)
        except:
            pass
