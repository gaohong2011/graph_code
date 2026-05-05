# Graph Code

English | [中文](#中文)

Graph Code is a LangGraph-based coding agent inspired by Claude Code and Codex CLI. It uses LangGraph for graph runtime, checkpointing, interrupts, streaming, and subgraph support instead of reimplementing those pieces.

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

This writes `docs/stategraph-topology.mmd` via `build_agent().get_graph().draw_mermaid()` and `docs/stategraph-topology.png` via `draw_mermaid_png()`.

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

See [docs/architecture.md](docs/architecture.md) and [docs/stategraph-structure.md](docs/stategraph-structure.md) for the graph, tool pipeline, and persistent entity model.

# 中文

[English](#graph-code) | 中文

Graph Code 是一个基于 LangGraph 的 coding agent，设计目标类似 Claude Code 和 Codex CLI。项目把 graph runtime、checkpoint、interrupt、streaming、subgraph 等能力交给 LangGraph，而不是在本项目里重复实现这些运行时能力。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 配置

真实模型使用 OpenAI-compatible API：

```bash
export LLM_API_KEY=sk-your-key
export LLM_BASE_URL=https://api.openai.com/v1
export LLM_MODEL=gpt-4o-mini
```

常用可选配置：

```bash
export WORKING_DIR=/path/to/project
export PERMISSION_MODE=default   # default, plan, auto
export MAX_TOOL_ITERATIONS=10
export OUTPUT_LIMIT=12000
export CHECKPOINT_BACKEND=memory  # memory, sqlite, postgres
export STORE_BACKEND=memory       # memory, postgres
```

`LLM_MODEL=mock` 会使用内置 mock model，不需要 API key。

## 运行

交互模式：

```bash
python -m graph_code --mock
python -m graph_code --model gpt-4o-mini
```

单次命令：

```bash
python -m graph_code --mock "hello"
python -m graph_code --permission-mode auto "read README.md and summarize it"
```

通过相同 LangGraph thread 恢复会话：

```bash
python -m graph_code --thread-id my-session --mock "first turn"
python -m graph_code --thread-id my-session --mock "continue"
```

Streaming 模式：

```bash
python -m graph_code --mock --stream-mode updates "hello"
python -m graph_code --mock --stream-mode updates,messages,custom "hello"
```

## 权限模式

- `default`：只读工具直接运行；写文件、bash、worktree、未知工具和 MCP 调用会请求确认。
- `plan`：有副作用的工具都请求确认，适合规划和审查阶段。
- `auto`：没有命中 deny rule 的工具自动运行；致命 deny rule 仍会阻止执行。

`bash` 会单独检查危险模式。`rm -rf /`、`dd if=`、fork bomb 等致命模式会被拒绝并作为 tool result 返回。`sudo`、`rm -rf`、命令替换、重定向、shell 控制符组合等可疑模式会在 `default` 和 `plan` 模式下触发 LangGraph `interrupt()` 审批。

## 工具

所有内置工具调用都会经过统一的 Tool Router、Permission Gate、pre/post hook、执行运行时和 `ToolResultEnvelope` 回写路径：

- 文件和搜索：`read_file`, `write_file`, `edit_file`, `search_files`
- 执行：`bash`
- 规划和上下文：`todo`, `load_skill`, `compact`, `save_memory`
- 任务：`task_create`, `task_update`, `task_get`, `task_list`, `task_complete`, `claim_task`
- 后台任务：`background_run`, `background_check`
- 调度：`schedule_create`, `schedule_list`, `schedule_delete`
- 团队协议：`team_spawn`, `send_message`, `request_shutdown`, `submit_plan_approval`
- Worktree：`worktree_create`, `worktree_enter`, `worktree_run`, `worktree_closeout`
- MCP 路由：工具名格式为 `mcp__{server}__{tool}`

大工具输出会写入 `.agent/tool-outputs/`；对话上下文只保留 preview 和 `[persisted-output: ...]` 标记。

## Mock 测试

```bash
python -m pytest -q
python -m graph_code --mock "hello"
```

## 图结构导出

StateGraph 拓扑图从编译后的 LangGraph graph 自动生成：

```bash
python -m graph_code.utils.export_graph_diagram --png
```

该命令通过 `build_agent().get_graph().draw_mermaid()` 写入 `docs/stategraph-topology.mmd`，并通过 `draw_mermaid_png()` 写入 `docs/stategraph-topology.png`。

## 真实模型运行

```bash
export LLM_API_KEY=sk-your-key
export LLM_BASE_URL=https://api.openai.com/v1
export LLM_MODEL=gpt-4o-mini
python -m graph_code --thread-id real-run-1 "inspect this repository"
```

## 持久化

开发默认使用 `InMemorySaver` 和 `InMemoryStore` 编译 graph：

```python
graph.compile(checkpointer=InMemorySaver(), store=InMemoryStore())
```

每次 invoke/stream 都使用：

```python
{"configurable": {"thread_id": "..."}}
```

代码结构允许生产部署把 memory checkpointer/store 替换为 SQLite 或 Postgres LangGraph 后端，而不需要修改节点逻辑。

后端通过环境变量选择：

```bash
# 开发默认
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

当前限制：仓库没有 vendored SQLite/Postgres LangGraph 后端包。未安装可选包时，启动会抛出明确错误并提示缺失包名。本地测试覆盖 memory 模式和缺失依赖错误；生产数据库迁移、连接池大小等需要在目标环境验证。

## MCP SDK

Mock MCP transport 已内置。真实 MCP transport 通过可选官方 Python SDK 接入：

```bash
pip install "mcp[cli]"
```

Manifest 示例：

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

Registry 支持 `stdio`、`streamable-http` 和 `http` transport。缺少 bearer token 时 server 状态为 `needs-auth`；缺少 SDK 依赖时状态为 `failed`。本仓库尚未实现 OAuth；可以先使用 bearer header，或在 `graph_code/mcp/client.py` 中接入 SDK auth provider。

## Worktree

当 base path 是真实 Git work tree 时，`worktree_create` 会调用 `git worktree add -b ...`。非 Git 目录会退化为 registry 目录。`worktree_closeout(remove)` 删除前会检查 dirty state，并对真实 worktree 使用 `git worktree remove`。

## 真实模型验证

支持 OpenAI-compatible API。Moonshot/Kimi 示例：

```bash
export LLM_API_KEY=...
export LLM_BASE_URL=https://api.moonshot.cn/v1
export LLM_MODEL=kimi-k2.5
python -m graph_code --thread-id real-smoke "请只回答 pong"
```

不要提交 `.env` 或 API key。`.env.example` 只能放 placeholder。

## 更多文档

查看 [docs/architecture.md](docs/architecture.md) 和 [docs/stategraph-structure.md](docs/stategraph-structure.md)，了解 graph、tool pipeline 和持久化实体模型。
