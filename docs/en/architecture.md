# Graph Code Architecture

[中文](../zh-CN/architecture.md)

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
