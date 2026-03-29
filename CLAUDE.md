# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Graph Code is a LangGraph-based AI programming assistant that provides an interactive CLI for coding tasks. It uses a state machine architecture where an LLM agent iteratively calls tools (file operations, code search, command execution) to complete user requests.

## Running the Application

### Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration
Required environment variables (or use `.env` file):
- `LLM_API_KEY` - API key for LLM service
- `LLM_BASE_URL` - API base URL (e.g., `https://api.moonshot.cn/v1`)
- `LLM_MODEL` - Model name (default: `gpt-4o-mini`)

Optional:
- `WORKING_DIR` - Working directory (default: current directory)
- `AUTO_CONFIRM` - Skip confirmations (default: `false`)
- `MAX_TOOL_ITERATIONS` - Safety limit (default: `10`)

### Run Commands
```bash
# Interactive mode
python -m graph_code

# Single command mode
python -m graph_code "list all Python files"

# With options
python -m graph_code --model moonshot-v1-8k --working-dir /path/to/project
```

## Architecture

### LangGraph State Machine

The agent is built as a LangGraph with three nodes:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   agent     │────▶│   tools     │────▶│   agent     │──┐
│   (llm)     │     │ (execution) │     │   (llm)     │  │
└─────────────┘     └─────────────┘     └─────────────┘  │
        │                                                │
        └────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────┐
│ check_interaction│──▶ END (pause for user input)
└─────────────────┘
```

**Flow:**
1. `agent_node` - LLM decides next action (tool call or response)
2. `tools_node` - Executes tool calls, results feed back to agent
3. `check_interaction_node` - Pauses for user confirmation/questions
4. `should_continue` - Routes to next node based on state

### Key Files

| File | Purpose |
|------|---------|
| `graph_code/agent/graph.py` | Builds and runs the LangGraph |
| `graph_code/agent/nodes.py` | Node implementations (agent, tools, interaction) |
| `graph_code/agent/state.py` | State TypedDict definition |
| `graph_code/tools/*.py` | Tool implementations |
| `graph_code/llm/client.py` | LLM client factory |
| `graph_code/config.py` | Configuration management |
| `graph_code/main.py` | CLI entry point |

### State Management

State (`GraphCodeState`) flows between nodes and includes:
- `messages` - Conversation history (LangChain message types)
- `tool_calls` - Pending tool calls from LLM
- `tool_results` - Results from executed tools
- `iteration_count` - Prevents infinite loops
- `pending_question` / `pending_confirmation` - Human-in-the-loop state

### Tools

Tools are LangChain `StructuredTool` instances defined in `graph_code/tools/`:

- **File tools** (`file_tools.py`): `read_file`, `write_file`, `list_directory`, `glob_search`
- **Code tools** (`code_tools.py`): `grep_search`, `read_code_chunk`
- **Exec tools** (`exec_tools.py`): `bash_command`, `python_execute`
- **Interaction tools** (`interaction.py`): `ask_user`, `confirm_action`

All file operations enforce security via `_get_safe_path()` which prevents access outside `WORKING_DIR`.

### LLM Client

Uses `langchain-openai`'s `ChatOpenAI` with base URL configuration to support any OpenAI-compatible API (Moonshot, OpenAI, Azure, Ollama, etc.).

Special handling for Kimi models in `llm/client.py`:
- Sets `temperature=1.0` for Kimi models
- Disables thinking mode for `k2.5` models to avoid API issues

### Security Model

- All file paths are resolved and checked to be within `WORKING_DIR`
- Destructive operations (write, bash) require user confirmation unless `AUTO_CONFIRM=true`
- Dangerous bash commands (rm -rf /, format, etc.) are blocked
- Tools can only access files within the configured working directory

## Adding New Tools

To add a tool:

1. Create the tool function in the appropriate `tools/` module or a new one
2. Wrap with `@lc_tool` decorator from `langchain_core.tools`
3. Add to `get_tools()` in `graph_code/agent/nodes.py`
4. Import the tool function in `nodes.py` if defined elsewhere

Example pattern from `nodes.py`:
```python
@lc_tool
def _read_file(file_path: str, offset: int = 0, limit: int = None):
    """Read content from a file."""
    return read_file(file_path, offset, limit)
```

## Testing Changes

There are no configured test runners. Test manually:

```bash
# Run the module
python -m graph_code

# Test a specific task
python -m graph_code "read the first 10 lines of README.md"
```

## Dependencies

Key dependencies from `requirements.txt`:
- `langgraph>=0.2.0` - Graph orchestration
- `langchain-core>=0.3.0` - Core abstractions
- `langchain-openai>=0.2.0` - OpenAI-compatible LLM client
- `rich>=13.0.0` - CLI UI and formatting
