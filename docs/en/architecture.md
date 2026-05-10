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

- Identity
- Task behavior
- Tool behavior
- Context behavior
- Project instructions
- Memory
- Environment

Static prompt sections can be cached and reused, but dynamic sections are rebuilt as needed. Project instructions, memory content, and environment details are refreshed so a compacted or resumed run sees current instructions, current memory, and current workspace context rather than a permanently frozen prompt.

Project instructions follow the Claude Code-style layout. The prompt builder loads `CLAUDE.md`, `.claude/CLAUDE.md`, and `.claude/rules/*.md` from the active workspace and presents them in the project-instructions section.

Long-term memory is file-based by default. Each project gets a memory root under `~/.graph-code/projects/<project-key>/memory/`, unless `GRAPH_CODE_MEMORY_DIR` overrides it. `MEMORY.md` acts as the index, and topic files include frontmatter with `name`, `description`, `type`, and `updated_at`. File tools validate runtime paths against the active workspace and the configured memory root, so memory files can be read and updated without broad filesystem access.

Session memory is optional and disabled by default. When `ENABLE_SESSION_MEMORY=true`, Graph Code can maintain `~/.graph-code/projects/<project-key>/session-memory/session.md`; when available, that session memory can be preferred as a compact source before falling back to model or extractive summaries. Memory relevance and automatic memory extraction are also opt-in, controlled by `ENABLE_MEMORY_RELEVANCE` and `ENABLE_AUTO_MEMORY_EXTRACTION`.

Skill bodies are not loaded into the prompt by default. The model sees the manifest and calls `load_skill` when it needs a full body.

## Context Compaction

The graph supports micro, summary, and manual compaction. The full transcript stays in `AgentState.messages`, while model calls use a separate `context_messages` list. This keeps LangGraph checkpoints and transcripts intact while shrinking the context actually sent to the model.

Compaction is layered:

1. `build_prompt` runs context management before every model call, so long no-tool histories are compacted before the provider request.
2. Token policy estimates current context size and records `compact_state.token_budget` plus warning state.
3. Micro compact first replaces old bulky compactable `ToolMessage` content while preserving assistant `tool_calls` / `ToolMessage.tool_call_id` protocol pairs. Read/search/bash/worktree/MCP results are compactable; protocol/team/control tools are kept intact.
4. Time-based micro compact can clear old compactable tool results after a configured turn gap.
5. If the context is still above the summary threshold, or the compatibility message-count threshold is reached, the graph creates a compact boundary plus a structured summary.
6. Recent messages are kept verbatim after the summary; retained boundaries are protocol-grouped so a tool-call group is not cut in half.
7. Manual compact requests from the `compact` tool are stored in `compact_state.pending_manual_request` and run through the same `compact_check` path.
8. Context-too-long provider errors trigger a reactive compact retry while `recovery_state["context_retry_budget"]` remains.

Summary compaction preserves:

- Current goal
- Completed actions
- Key files
- Key decisions
- Next step

With a real model configuration, summary compact attempts one no-tools summarizer call using a Claude Code-style section prompt. If the summarizer prompt is too long, it retries once with a shorter prompt. If summarization still fails, the graph falls back to the local extractive summary so compaction does not interrupt the main loop. Consecutive summarizer failures trip a circuit breaker.

Summary compact writes the full pre-compact transcript to `.agent/transcripts/{boundary}.jsonl`, runs optional `.agent/hooks/pre_compact.py` and `.agent/hooks/post_compact.py`, and rehydrates current task, planning state, loaded skill manifest, worktree context, MCP connection state, notifications, and transcript path into post-compact context.

Large tool output is not kept permanently in model context; persisted output markers point to disk. Tune behavior with `CONTEXT_WINDOW_TOKENS`, `MICRO_COMPACT_RATIO`, `AUTO_COMPACT_RATIO`, `COMPACT_RECENT_MESSAGES`, `MICRO_COMPACT_KEEP_TOOL_RESULTS`, `COMPACT_MESSAGE_COUNT_THRESHOLD`, `COMPACT_USE_MODEL_SUMMARY`, `COMPACT_WARNING_RATIO`, `COMPACT_FAILURE_CIRCUIT_BREAKER`, `COMPACT_SUMMARY_RETRY_BUDGET`, and `TIME_BASED_MICROCOMPACT_TURN_GAP`.

## Recovery

`recovery_state` tracks separate budgets for:

- Max-token continuation
- Context-too-long compact retry
- Transient API/network retry

Each recovery path writes `transition_reason` so the CLI and tests can inspect why the graph moved.

## Persistent Entities

Task, runtime, schedule, team, request, worktree, and tool-output records live under `.agent/`.

```text
.agent/
  tasks/
  runtime-tasks/
  schedules/
  teams/
  requests/
  worktrees/
  tool-outputs/
```

Long-term memory is stored separately under `~/.graph-code/projects/<project-key>/memory/`, or under `GRAPH_CODE_MEMORY_DIR` when that override is set.

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
