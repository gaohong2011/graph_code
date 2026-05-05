# Graph Code

Graph Code is a LangGraph-based coding agent inspired by Claude Code and Codex CLI. It uses LangGraph for the graph runtime, checkpointing, interrupts, streaming, and subgraph support instead of reimplementing those pieces.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configure

For a real OpenAI-compatible model:

```bash
export LLM_API_KEY=sk-your-key
export LLM_BASE_URL=https://api.openai.com/v1
export LLM_MODEL=gpt-4o-mini
```

Useful optional settings:

```bash
export WORKING_DIR=/path/to/project
export PERMISSION_MODE=default   # default, plan, auto
export MAX_TOOL_ITERATIONS=10
export OUTPUT_LIMIT=12000
export CHECKPOINT_BACKEND=memory  # memory, sqlite, postgres
export STORE_BACKEND=memory       # memory, postgres
```

`LLM_MODEL=mock` uses the built-in mock model and does not require an API key.

## Run

Interactive mode:

```bash
python -m graph_code --mock
python -m graph_code --model gpt-4o-mini
```

Single command:

```bash
python -m graph_code --mock "hello"
python -m graph_code --permission-mode auto "read README.md and summarize it"
```

Resume a conversation by reusing the same LangGraph thread:

```bash
python -m graph_code --thread-id my-session --mock "first turn"
python -m graph_code --thread-id my-session --mock "continue"
```

Streaming modes:

```bash
python -m graph_code --mock --stream-mode updates "hello"
python -m graph_code --mock --stream-mode updates,messages,custom "hello"
```

## Permission Modes

- `default`: read-only tools run directly; writes, bash, worktree operations, unknown tools, and MCP calls ask for approval.
- `plan`: side-effecting tools ask for approval, suitable for planning/review phases.
- `auto`: non-denied tools run without asking; fatal deny rules still block execution.

Dangerous bash patterns are checked separately. Fatal patterns such as `rm -rf /`, `dd if=`, and fork bombs are denied and returned as tool results. Suspicious patterns such as `sudo`, `rm -rf`, command substitution, redirection, and shell control operators trigger LangGraph `interrupt()` approval in `default` and `plan` modes.

## Tools

Built-in tool calls all pass through the same Tool Router, Permission Gate, pre/post hook points, execution runtime, and `ToolResultEnvelope` response path:

- Files/search: `read_file`, `write_file`, `edit_file`, `search_files`
- Execution: `bash`
- Planning/context: `todo`, `load_skill`, `compact`, `save_memory`
- Tasks: `task_create`, `task_update`, `task_get`, `task_list`, `task_complete`, `claim_task`
- Background: `background_run`, `background_check`
- Schedules: `schedule_create`, `schedule_list`, `schedule_delete`
- Team protocol: `team_spawn`, `send_message`, `request_shutdown`, `submit_plan_approval`
- Worktrees: `worktree_create`, `worktree_enter`, `worktree_run`, `worktree_closeout`
- MCP routing: tool names shaped as `mcp__{server}__{tool}`

Large tool outputs are written to `.agent/tool-outputs/`; the conversation keeps a preview plus a `[persisted-output: ...]` marker.

## Mock Tests

```bash
python -m pytest -q
python -m graph_code --mock "hello"
```

## Graph Diagram

The StateGraph topology image is generated from the compiled LangGraph graph:

```bash
python -m graph_code.utils.export_graph_diagram --png
```

This writes `docs/stategraph-topology.mmd` via
`build_agent().get_graph().draw_mermaid()` and `docs/stategraph-topology.png`
via `draw_mermaid_png()`.

## Real Model Run

```bash
export LLM_API_KEY=sk-your-key
export LLM_BASE_URL=https://api.openai.com/v1
export LLM_MODEL=gpt-4o-mini
python -m graph_code --thread-id real-run-1 "inspect this repository"
```

## Persistence

The development build compiles the graph with `InMemorySaver` and `InMemoryStore`:

```python
graph.compile(checkpointer=InMemorySaver(), store=InMemoryStore())
```

Every invoke/stream call uses:

```python
{"configurable": {"thread_id": "..."}}
```

The code is structured so production deployments can replace the memory checkpointer/store with SQLite or Postgres-backed LangGraph backends without changing node logic.

Backend selection is controlled by environment variables:

```bash
# Development default
export CHECKPOINT_BACKEND=memory
export STORE_BACKEND=memory

# SQLite checkpoints
pip install langgraph-checkpoint-sqlite
export CHECKPOINT_BACKEND=sqlite
export CHECKPOINT_URI=.agent/checkpoints.sqlite

# Postgres checkpoints/store
pip install langgraph-checkpoint-postgres
export CHECKPOINT_BACKEND=postgres
export CHECKPOINT_URI=postgresql://user:pass@localhost:5432/graph_code
export STORE_BACKEND=postgres
export STORE_URI=postgresql://user:pass@localhost:5432/graph_code
```

Current limitation: the repository does not vendor SQLite/Postgres LangGraph backend packages. If those optional packages are not installed, startup raises a clear error naming the missing package. The local test suite covers memory mode and missing-dependency errors; production database migrations and pool sizing must be verified in the target deployment environment.

## MCP SDK

Mock MCP transport is built in. Real MCP transports are wired through the optional official Python SDK:

```bash
pip install "mcp[cli]"
```

Manifest examples:

```json
{
  "servers": {
    "fs": {
      "transport": "stdio",
      "command": "python",
      "args": ["server.py"]
    },
    "private": {
      "transport": "streamable-http",
      "url": "https://example.com/mcp",
      "auth": {
        "type": "bearer",
        "token_env": "MCP_PRIVATE_TOKEN"
      }
    }
  }
}
```

Real transports supported by the registry are `stdio`, `streamable-http`, and `http`. Missing bearer tokens mark the server as `needs-auth`; missing SDK dependencies mark it as `failed`. OAuth is not implemented in this repository yet; use bearer headers or add an SDK auth provider in `graph_code/mcp/client.py`.

## Worktrees

`worktree_create` now uses `git worktree add -b ...` when the base path is a real Git work tree. For non-Git directories it falls back to a registry directory. `worktree_closeout(remove)` checks dirty state first and uses `git worktree remove` for real worktrees.

## Real Model Verification

Real model runs are supported with OpenAI-compatible APIs. Example Moonshot/Kimi setup:

```bash
export LLM_API_KEY=...
export LLM_BASE_URL=https://api.moonshot.cn/v1
export LLM_MODEL=kimi-k2.5
python -m graph_code --thread-id real-smoke "请只回答 pong"
```

Never commit `.env` or API keys. `.env.example` should contain placeholders only.

## More

See [docs/architecture.md](docs/architecture.md) for the graph, tool pipeline, and persistent entity model.
