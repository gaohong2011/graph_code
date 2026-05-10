# Graph Code 架构说明

[English](../en/architecture.md)

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

- Identity
- Task behavior
- Tool behavior
- Context behavior
- Project instructions
- Memory
- Environment

静态 prompt 分段可以缓存复用，但动态分段会按需重建。Project instructions、memory 内容和 environment 信息会刷新，因此压缩或恢复后的运行会看到当前 instructions、当前 memory 和当前工作区上下文，而不是永久冻结的 prompt。

Project instructions 使用 Claude Code 风格布局。Prompt builder 会从当前工作区加载 `CLAUDE.md`、`.claude/CLAUDE.md` 和 `.claude/rules/*.md`，并放入 project-instructions 分段。

长期 memory 默认基于文件。每个项目的 memory root 位于 `~/.graph-code/projects/<project-key>/memory/`，也可以通过 `GRAPH_CODE_MEMORY_DIR` 覆盖。`MEMORY.md` 是索引文件，topic 文件包含 `name`、`description`、`type` 和 `updated_at` frontmatter。文件工具会在运行时校验路径只能落在当前 workspace 或配置的 memory root 内，因此可以安全读写 memory 文件，而不会放开整个文件系统。

Session memory 是可选能力，默认关闭。设置 `ENABLE_SESSION_MEMORY=true` 后，Graph Code 可以维护 `~/.graph-code/projects/<project-key>/session-memory/session.md`；存在时，它可以作为优先的 compact 来源，然后再回退到模型 summary 或 extractive summary。Memory relevance 和 automatic memory extraction 也默认关闭，分别由 `ENABLE_MEMORY_RELEVANCE` 和 `ENABLE_AUTO_MEMORY_EXTRACTION` 控制。

Skill 正文默认不会加载进 prompt。模型先看到 manifest，需要完整正文时调用 `load_skill`。

## 上下文压缩

Graph 支持 micro、summary 和 manual compaction。完整 transcript 保留在 `AgentState.messages`，模型调用使用单独的 `context_messages`。这样既不破坏 LangGraph checkpoint / transcript，又能让模型实际看到更小的上下文。

压缩策略按层执行：

1. `build_prompt` 在每次模型调用前运行上下文管理，因此无工具的长对话也会在请求 provider 前压缩。
2. Token policy 先估算当前上下文大小，并记录 `compact_state.token_budget` 和 warning state。
3. Micro compact 优先压缩旧的可压缩 `ToolMessage` 大输出，保留最近的 tool result 原文，并且不改变 assistant `tool_calls` / `ToolMessage.tool_call_id` 配对。read/search/bash/worktree/MCP 结果可压缩，协议/team/control 工具结果保持原样。
4. Time-based micro compact 可以在配置的 turn gap 后清理旧的可压缩 tool result。
5. 如果 micro 后仍超过 summary 阈值，或者达到兼容的消息数阈值，则创建 compact boundary 和结构化 summary。
6. Summary 后保留最近消息原文；保留边界按协议分组，避免裁掉 tool-call group 的一半。
7. `compact` 工具发出的 manual compact request 会写入 `compact_state.pending_manual_request`，并通过同一个 `compact_check` 路径处理。
8. provider 返回 context-too-long 时，只要 `recovery_state["context_retry_budget"]` 还有预算，就触发 reactive compact retry。

Summary compaction 保留：

- 当前目标
- 已完成动作
- 关键文件
- 关键决策
- 下一步

真实模型配置下，summary compact 会用 Claude Code 风格的 section prompt 调用一次无工具 summarizer。summarizer 自己 prompt-too-long 时，会用更短 prompt 重试一次。仍失败时回退本地 extractive summary，避免压缩失败打断主流程。连续 summarizer 失败会触发 circuit breaker。

Summary compact 会把完整压缩前 transcript 写到 `.agent/transcripts/{boundary}.jsonl`，运行可选 `.agent/hooks/pre_compact.py` 和 `.agent/hooks/post_compact.py`，并在压缩后重新注入 current task、planning state、loaded skill manifest、worktree context、MCP connection state、notifications 和 transcript path。

大工具输出不会长期留在模型上下文；persisted output marker 会指向磁盘文件。可通过 `CONTEXT_WINDOW_TOKENS`、`MICRO_COMPACT_RATIO`、`AUTO_COMPACT_RATIO`、`COMPACT_RECENT_MESSAGES`、`MICRO_COMPACT_KEEP_TOOL_RESULTS`、`COMPACT_MESSAGE_COUNT_THRESHOLD`、`COMPACT_USE_MODEL_SUMMARY`、`COMPACT_WARNING_RATIO`、`COMPACT_FAILURE_CIRCUIT_BREAKER`、`COMPACT_SUMMARY_RETRY_BUDGET` 和 `TIME_BASED_MICROCOMPACT_TURN_GAP` 调整策略。

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
