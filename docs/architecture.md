# Graph Code Architecture / 架构说明

English | [中文](#中文)

For a node-by-node view of the main LangGraph runtime, see [`stategraph-structure.md`](stategraph-structure.md).

## LangGraph Query Graph

The main agent is a `StateGraph(AgentState)` compiled with a checkpointer and store. The graph uses `START`, `END`, `add_node`, `add_edge`, and `add_conditional_edges`.

```text
START
  -> drain_notifications
  -> build_prompt
  -> call_model
  -> recovery_handler
  -> route_model_response
      -> retry -> call_model
      -> final_response -> END
      -> permission_gate
          -> human_permission_interrupt
              -> permission_gate
          -> run_pre_tool_hooks
          -> execute_tools
          -> run_post_tool_hooks
          -> append_tool_results
          -> compact_check
          -> recovery_handler_after_tools
          -> call_model
```

`AgentState.messages` uses LangGraph's `add_messages` reducer. The rest of the state is explicit control-plane data: pending tool calls, permission requests, tool envelopes, planning, compact state, recovery budgets, loaded skills, notifications, runtime tasks, teammate identity, worktree context, and MCP connection state.

Every stream/invoke path supplies:

```python
{"configurable": {"thread_id": thread_id}}
```

This lets LangGraph resume the same thread after checkpoints and human interrupts.

Persistence backends are created through `graph_code.agent.persistence`. The default is `InMemorySaver` plus `InMemoryStore`; optional production selections are SQLite checkpoints and Postgres checkpoints/store, gated by the corresponding LangGraph backend packages.

## Tool Pipeline

The execution path is:

```text
LLM tool_call
  -> Tool Router
  -> Permission Gate
  -> optional interrupt()
  -> PreToolUse hooks
  -> ToolExecutionRuntime
  -> PostToolUse hooks
  -> ToolResultEnvelope
  -> ToolMessage
```

The permission gate evaluates in this order:

1. Deny rules
2. Mode check
3. Allow rules
4. Ask user

`human_permission_interrupt` uses LangGraph `interrupt()` and resumes with `Command(resume=...)`. Denials are written back as normal tool results, so the graph does not crash.

Read-only tools can run concurrently. Write tools, bash, and worktree operations are serialized. Results are always returned in the original tool-call order. Large outputs are persisted under `.agent/tool-outputs/` and represented in context with a preview plus a persisted-output marker.

## Prompt, Memory, Skills

The prompt is assembled from stable sections:

- Core behavior
- Tool manifest
- Skills manifest
- Long-term memory
- Project instructions
- Dynamic context

Skill bodies are not loaded into the prompt by default. The model sees the manifest and calls `load_skill` when it needs a full body. Long-term memory is addressed by namespace/key and stored under `.agent/memory/` in the local runtime; the graph compile path also provides a LangGraph store for production backends.

## Context Compaction

The graph supports micro, summary, and manual compaction. Summary compaction preserves:

- Current goal
- Completed actions
- Key files
- Key decisions
- Next step

Large tool output is not kept permanently in `messages`; persisted output markers point to disk.

## Recovery

`recovery_state` tracks separate budgets for:

- Max-token continuation
- Context-too-long compact retry
- Transient API/network retry

Each recovery path writes `transition_reason` so the CLI and tests can inspect why the graph moved.

## Persistent Entities

All durable local records live under `.agent/`.

```text
.agent/
  tasks/
  runtime-tasks/
  schedules/
  teams/
  requests/
  worktrees/
  tool-outputs/
  memory/
```

Task records contain `id`, `subject`, `description`, `status`, `blockedBy`, `blocks`, `owner`, and `worktree`. Completing a task removes it from downstream `blockedBy` lists and unlocks dependents when no blockers remain. Claiming uses a lock file and appends a claim event.

Runtime tasks persist the command, pid, status, output path, return code, and timestamps. `background_run` returns immediately with a runtime task id. `background_check` turns completion into a notification.

Schedules contain `id`, `cron`, `prompt`, `recurring`, `durable`, `created_at`, and `last_fired_at`. Due schedules enqueue notifications only; they do not execute prompts directly.

Teammates have `name`, `role`, `status`, `thread_id`, and an inbox. Protocol messages carry a `request_id` and are persisted as `RequestRecord` files. Shutdown requests and plan approvals use the same request record path.

Worktree records separate what to do from where to do it. `worktree_create` calls `git worktree add -b ...` when the base path is a real Git work tree, and falls back to a registry directory for non-Git paths. `worktree_closeout(remove)` checks dirty state before deleting and raises instead of discarding changes.

## MCP And Plugins

`MCPClientRegistry` loads plugin/MCP manifests and maps tools as:

```text
mcp__{server}__{tool}
```

MCP calls return `ToolResultEnvelope` and use the same permission and tool-result pipeline as built-ins. Connection state values include `connected`, `pending`, `failed`, `needs-auth`, and `disabled`. Mock transport is built in for deterministic tests. Real `stdio`, `streamable-http`, and `http` transports are wired through the optional official MCP Python SDK, imported lazily so local mock/test runs do not require the package.

## Subagents

One-shot subagents are implemented as a LangGraph subgraph in `graph_code.agent.subagents`. They maintain isolated messages and return only a compact summary. Long-lived teammates are represented as independent records with separate thread ids and inboxes.

# 中文

[English](#graph-code-architecture--架构说明) | 中文

主 LangGraph runtime 的逐节点说明见 [`stategraph-structure.md`](stategraph-structure.md)。

## LangGraph 查询图

主 agent 是一个带 checkpointer 和 store 编译的 `StateGraph(AgentState)`。图结构使用 `START`、`END`、`add_node`、`add_edge` 和 `add_conditional_edges`。

```text
START
  -> drain_notifications
  -> build_prompt
  -> call_model
  -> recovery_handler
  -> route_model_response
      -> retry -> call_model
      -> final_response -> END
      -> permission_gate
          -> human_permission_interrupt
              -> permission_gate
          -> run_pre_tool_hooks
          -> execute_tools
          -> run_post_tool_hooks
          -> append_tool_results
          -> compact_check
          -> recovery_handler_after_tools
          -> call_model
```

`AgentState.messages` 使用 LangGraph 的 `add_messages` reducer。其他 state 字段是显式控制面数据，包括 pending tool calls、权限请求、tool envelope、planning、compact state、recovery budget、已加载 skill、通知、runtime task、teammate identity、worktree context 和 MCP connection state。

每条 stream/invoke 路径都传入：

```python
{"configurable": {"thread_id": thread_id}}
```

这样 LangGraph 可以在 checkpoint 和 human interrupt 之后恢复同一个 thread。

持久化后端通过 `graph_code.agent.persistence` 创建。默认是 `InMemorySaver` 加 `InMemoryStore`；生产可选 SQLite checkpoint 和 Postgres checkpoint/store，对应依赖由 LangGraph 后端包提供。

## 工具执行管线

执行路径如下：

```text
LLM tool_call
  -> Tool Router
  -> Permission Gate
  -> optional interrupt()
  -> PreToolUse hooks
  -> ToolExecutionRuntime
  -> PostToolUse hooks
  -> ToolResultEnvelope
  -> ToolMessage
```

Permission Gate 的判断顺序：

1. Deny rules
2. Mode check
3. Allow rules
4. Ask user

`human_permission_interrupt` 使用 LangGraph `interrupt()` 暂停，并通过 `Command(resume=...)` 恢复。拒绝会作为普通 tool result 回写，因此 graph 不会崩溃。

只读工具可以并发运行。写工具、bash 和 worktree 操作会串行执行。结果始终按原始 tool-call 顺序返回。大输出会持久化到 `.agent/tool-outputs/`，上下文里只保留 preview 和 persisted-output 标记。

## Prompt、Memory、Skills

Prompt 由稳定分段组成：

- Core behavior
- Tool manifest
- Skills manifest
- Long-term memory
- Project instructions
- Dynamic context

Skill 正文默认不会加载进 prompt。模型先看到 manifest，需要完整正文时调用 `load_skill`。长期 memory 通过 namespace/key 访问，本地 runtime 存在 `.agent/memory/`；graph compile 路径也提供 LangGraph store，供生产后端替换。

## 上下文压缩

Graph 支持 micro、summary 和 manual compaction。Summary compaction 保留：

- 当前目标
- 已完成动作
- 关键文件
- 关键决策
- 下一步

大工具输出不会长期留在 `messages`；persisted output marker 会指向磁盘文件。

## 错误恢复

`recovery_state` 分别跟踪以下预算：

- Max-token continuation
- Context-too-long compact retry
- Transient API/network retry

每条恢复路径都会写入 `transition_reason`，便于 CLI 和测试检查 graph 转移原因。

## 持久化实体

所有本地持久化记录都在 `.agent/` 下：

```text
.agent/
  tasks/
  runtime-tasks/
  schedules/
  teams/
  requests/
  worktrees/
  tool-outputs/
  memory/
```

Task record 包含 `id`、`subject`、`description`、`status`、`blockedBy`、`blocks`、`owner` 和 `worktree`。任务完成后会从下游任务的 `blockedBy` 中移除并在无阻塞时解锁依赖任务。Claim 使用锁文件，并追加 claim event。

Runtime task 持久化 command、pid、status、output path、return code 和时间戳。`background_run` 会立即返回 runtime task id。`background_check` 会把完成状态转换为 notification。

Schedule 包含 `id`、`cron`、`prompt`、`recurring`、`durable`、`created_at` 和 `last_fired_at`。到期 schedule 只入队 notification，不直接执行 prompt。

Teammate 包含 `name`、`role`、`status`、`thread_id` 和 inbox。协议消息携带 `request_id`，并持久化为 `RequestRecord` 文件。Shutdown request 和 plan approval 走同一套 request record 路径。

Worktree record 把“做什么”和“在哪里做”分开。`worktree_create` 在 base path 是真实 Git work tree 时调用 `git worktree add -b ...`，非 Git 路径退化为 registry 目录。`worktree_closeout(remove)` 删除前检查 dirty state，不会丢弃改动。

## MCP 和插件

`MCPClientRegistry` 加载 plugin/MCP manifest，并把工具映射为：

```text
mcp__{server}__{tool}
```

MCP 调用返回 `ToolResultEnvelope`，并和内置工具使用同一套 permission 与 tool-result pipeline。连接状态包括 `connected`、`pending`、`failed`、`needs-auth` 和 `disabled`。Mock transport 已内置，方便确定性测试。真实 `stdio`、`streamable-http` 和 `http` transport 通过可选官方 MCP Python SDK 接入，并延迟 import，因此本地 mock/test 不需要安装该包。

## Subagents

一次性 subagent 在 `graph_code.agent.subagents` 中实现为 LangGraph subgraph。它们维护隔离 messages，并只返回紧凑摘要。长期 teammate 以独立记录表示，拥有自己的 thread id 和 inbox。
