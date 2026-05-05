# StateGraph Structure / StateGraph 结构

English / 中文

This document describes the main LangGraph graph implemented in `graph_code.agent.graph.build_agent()`. It is the authoritative map for the agent loop, routing decisions, interrupt/resume behavior, and tool-result message flow.

本文说明 `graph_code.agent.graph.build_agent()` 实现的主 LangGraph 图。它是 agent loop、路由决策、interrupt/resume 行为和 tool-result message 回流的权威说明。

## Source Files / 源文件

- `graph_code/agent/graph.py`: builds and compiles the `StateGraph`; 构建并编译 `StateGraph`。
- `graph_code/agent/state.py`: defines `AgentState`; 定义 `AgentState`。
- `graph_code/agent/nodes.py`: implements graph nodes and route functions; 实现图节点和路由函数。
- `graph_code/main.py`: drives `graph.stream(...)` and `Command(resume=...)` from the CLI; 在 CLI 中驱动 `graph.stream(...)` 和 `Command(resume=...)`。

## State Type / 状态类型

The graph is declared as:

图声明如下：

```python
workflow = StateGraph(AgentState)
```

`AgentState.messages` uses LangGraph's `add_messages` reducer. This keeps assistant tool calls and matching `ToolMessage` results in the message history without manual duplication.

`AgentState.messages` 使用 LangGraph 的 `add_messages` reducer。这样 assistant tool call 和对应 `ToolMessage` 会保留在 message history 中，不需要手工重复拼接。

The rest of `AgentState` is control-plane state:

`AgentState` 其余字段是控制面状态：

```text
messages
turn_count
transition_reason
pending_tool_calls
approved_tool_calls
pending_permission_request
tool_results
planning_state
compact_state
recovery_state
loaded_skills
notifications
runtime_tasks
current_task_id
teammate_identity
worktree_context
mcp_connection_state
```

Compatibility fields such as `tool_calls`, `iteration_count`, `pending_question`, `pending_confirmation`, `final_response`, and `error` are kept for the existing public API and tests.

兼容字段如 `tool_calls`、`iteration_count`、`pending_question`、`pending_confirmation`、`final_response` 和 `error` 会继续保留，用于现有 public API 和测试。

## Compile-Time Dependencies / 编译依赖

The graph is compiled with both a checkpointer and store:

Graph 编译时同时注入 checkpointer 和 store：

```python
workflow.compile(
    checkpointer=checkpointer or _default_checkpointer(cfg),
    store=store or _default_store(cfg),
)
```

The development default is memory-backed persistence. Non-memory backends are created through `graph_code.agent.persistence`, leaving SQLite/Postgres paths behind configuration.

开发默认使用 memory-backed persistence。非 memory 后端通过 `graph_code.agent.persistence` 创建，并通过配置预留 SQLite/Postgres 路径。

Every invoke/stream/resume path must pass a thread id:

每次 invoke/stream/resume 都必须传入 thread id：

```python
{"configurable": {"thread_id": thread_id}}
```

LangGraph uses this id to resume the same checkpoint after interrupts.

LangGraph 使用该 id 在 interrupt 后恢复同一个 checkpoint。

## Node Inventory / 节点清单

| Node | Function | Responsibility / 职责 |
| --- | --- | --- |
| `drain_notifications` | `drain_notifications` | Marks queued notifications before prompt/model work. / 在 prompt/model 前标记待处理通知。 |
| `build_prompt` | `build_prompt` | Prompt assembly hook; current implementation records `prompt_built`. / Prompt 组装 hook；当前实现记录 `prompt_built`。 |
| `call_model` | `call_model_node` -> `call_model` | Calls the configured chat model, validates tool-message protocol, or reuses pending tool calls. / 调用配置的 chat model，校验 tool-message 协议，或复用 pending tool calls。 |
| `recovery_handler` | `recovery_handler` | Handles model errors and transient retry routing before model response routing. / 在模型响应路由前处理模型错误和 transient retry。 |
| `route_model_response` | `lambda state: {}` | Empty node used as a named routing point after recovery. / 空节点，用作 recovery 之后的具名路由点。 |
| `permission_gate` | `permission_gate_node` -> `permission_gate` | Evaluates permissions and creates permission requests, denied envelopes, or executable calls. / 评估权限并创建审批请求、拒绝 envelope 或可执行调用。 |
| `human_permission_interrupt` | `human_permission_interrupt` | Calls LangGraph `interrupt()` and converts resume payloads into approved/denied tool state. / 调用 LangGraph `interrupt()`，并把 resume payload 转为 approved/denied tool state。 |
| `run_pre_tool_hooks` | `run_pre_tool_hooks` | Pre-tool hook point. / 工具执行前 hook 点。 |
| `execute_tools` | `execute_tools_node` -> `execute_tools` | Runs approved tool calls through `ToolExecutionRuntime`. / 通过 `ToolExecutionRuntime` 执行已批准工具调用。 |
| `run_post_tool_hooks` | `run_post_tool_hooks` | Post-tool hook point. / 工具执行后 hook 点。 |
| `append_tool_results` | `append_tool_results` | Converts `ToolResultEnvelope` records into `ToolMessage` objects. / 把 `ToolResultEnvelope` 转为 `ToolMessage`。 |
| `compact_check` | `compact_check` | Performs summary compaction when message history crosses the threshold. / 当 message history 超过阈值时执行 summary compaction。 |
| `recovery_handler_after_tools` | `recovery_handler` | Recovery hook after tool messages are appended and compaction runs. / tool message 回写和 compaction 后的 recovery hook。 |
| `final_response` | `final_response` | Produces the final user-visible response and ends the graph. / 生成用户可见最终响应并结束 graph。 |

## Graph Topology / 图拓扑

![Graph Code StateGraph topology](stategraph-topology.png)

The PNG above is generated from the compiled LangGraph object, not drawn by hand. Regenerate both the Mermaid source and PNG with:

上面的 PNG 来自编译后的 LangGraph 对象，不是手画图。重新生成 Mermaid 源和 PNG：

```bash
python -m graph_code.utils.export_graph_diagram --png
```

The exact Mermaid source produced by `build_agent().get_graph().draw_mermaid()` is stored in [`stategraph-topology.mmd`](stategraph-topology.mmd).

`build_agent().get_graph().draw_mermaid()` 生成的精确 Mermaid 源保存在 [`stategraph-topology.mmd`](stategraph-topology.mmd)。该文件中的节点名保持代码标识符，便于和真实 graph 对齐。

## Route Functions / 路由函数

### `route_model_response`

Runs after `recovery_handler`.

在 `recovery_handler` 之后运行。

| Return | Next node | Meaning / 含义 |
| --- | --- | --- |
| `retry` | `call_model` | A transient model/API error was recovered and should be retried within budget. / transient 模型/API 错误已恢复，应在预算内重试。 |
| `tools` | `permission_gate` | The model produced tool calls or pending tool calls already exist. / 模型产生了 tool calls，或已有 pending tool calls。 |
| `final` | `final_response` | The model produced normal content or an unrecoverable error exists. / 模型产生普通内容，或存在不可恢复错误。 |

### `route_permission`

Runs after `permission_gate`.

在 `permission_gate` 之后运行。

| Return | Next node | Meaning / 含义 |
| --- | --- | --- |
| `interrupt` | `human_permission_interrupt` | At least one tool call requires human approval. / 至少一个工具调用需要人工审批。 |
| `execute` | `run_pre_tool_hooks` | Tool calls are allowed or already approved. / 工具调用已允许或已批准。 |
| `append` | `append_tool_results` | Only denied tool results need to be returned to the model. / 只有被拒绝的 tool result 需要返回给模型。 |
| `final` | `final_response` | No tool work remains. / 没有剩余工具工作。 |

### `route_after_human_permission`

Runs after the graph resumes from `interrupt()`.

在 graph 从 `interrupt()` 恢复后运行。

| Return | Next node | Meaning / 含义 |
| --- | --- | --- |
| `permission` | `permission_gate` | More pending tool calls remain and must be checked separately. / 仍有 pending tool calls，需要单独检查。 |
| `execute` | `run_pre_tool_hooks` | Approved calls are ready to run. / 已批准调用可以执行。 |
| `append` | `append_tool_results` | Only denied tool results need to be appended. / 只需要追加被拒绝的 tool result。 |

This route prevents one approval from authorizing every side-effecting tool in the same model response.

该路由避免一次审批授权同一模型响应里的所有有副作用工具。

## Main Execution Paths / 主执行路径

### 1. Direct Final Response / 直接最终响应

```text
START
  -> drain_notifications
  -> build_prompt
  -> call_model
  -> recovery_handler
  -> route_model_response(final)
  -> final_response
  -> END
```

### 2. Tool Call Without Human Approval / 无人工审批的工具调用

Read-only tools and auto-approved tools follow:

只读工具和自动批准工具走这条路径：

```text
call_model
  -> route_model_response(tools)
  -> permission_gate(execute)
  -> run_pre_tool_hooks
  -> execute_tools
  -> run_post_tool_hooks
  -> append_tool_results
  -> compact_check
  -> recovery_handler_after_tools
  -> call_model
```

The second `call_model` sees the original assistant tool call plus matching `ToolMessage` entries and can produce a final response or request more tools.

第二次 `call_model` 会看到原始 assistant tool call 和匹配的 `ToolMessage`，然后可以生成最终响应或继续请求工具。

### 3. Tool Call With Human Approval / 需要人工审批的工具调用

Side-effecting tools in `default` or `plan` mode use LangGraph interrupts:

`default` 或 `plan` 模式下，有副作用工具使用 LangGraph interrupt：

```text
permission_gate(interrupt)
  -> human_permission_interrupt
      interrupt(permission_request)
      Command(resume={"approved": true})
  -> route_after_human_permission
```

Approved tool calls are stored in `approved_tool_calls`. If more pending tool calls remain, the graph routes back to `permission_gate` so each risky tool is checked independently. When no more approvals are needed, approved calls move to `execute_tools`.

已批准工具调用会存入 `approved_tool_calls`。如果还有 pending tool calls，graph 会回到 `permission_gate`，确保每个风险工具单独检查。无需更多审批时，已批准调用进入 `execute_tools`。

### 4. Tool Denial / 工具拒绝

If the user denies a permission request, `human_permission_interrupt` creates a normal error `ToolResultEnvelope` for that specific `tool_call_id`.

如果用户拒绝权限请求，`human_permission_interrupt` 会为对应 `tool_call_id` 创建普通 error `ToolResultEnvelope`。

Denied results do not crash the graph. They are appended as `ToolMessage` objects so the model receives a valid response for the denied tool call.

拒绝结果不会让 graph 崩溃。它们会作为 `ToolMessage` 追加，确保模型收到被拒绝工具调用的合法响应。

### 5. Transient Model Retry / transient 模型重试

When `call_model` catches a transient API/network error, it stores:

当 `call_model` 捕获 transient API/network error 时，会写入：

```text
error = "...provider error..."
transition_reason = "model_error"
```

`recovery_handler` checks `recovery_state["transient_retry_budget"]`. If budget remains, it clears `error`, decrements the budget, writes `transition_reason = "transient_model_retry"`, and `route_model_response` routes back to `call_model`.

`recovery_handler` 会检查 `recovery_state["transient_retry_budget"]`。如果还有预算，它会清空 `error`，递减预算，写入 `transition_reason = "transient_model_retry"`，然后由 `route_model_response` 路由回 `call_model`。

### 6. Context Compaction Loop / 上下文压缩循环

After tool results are appended, `compact_check` may summarize long history. The graph then passes through `recovery_handler_after_tools` and returns to `call_model`.

tool result 追加后，`compact_check` 可能会总结长历史。随后 graph 经过 `recovery_handler_after_tools` 并回到 `call_model`。

## Interrupt Payloads / Interrupt 载荷

`permission_gate` creates `pending_permission_request` using:

`permission_gate` 使用以下函数创建 `pending_permission_request`：

```python
build_permission_request(tool_call, decision)
```

The request contains:

请求包含：

```text
tool_call
tool_call_id
tool_name
args
reason
risk
```

`human_permission_interrupt` pauses with:

`human_permission_interrupt` 通过以下调用暂停：

```python
resume = interrupt(request)
```

The CLI resumes with:

CLI 通过以下方式恢复：

```python
Command(resume={"approved": True})
Command(resume={"approved": False, "reason": "..."})
```

## Message Protocol Invariants / Message 协议不变量

The graph must preserve OpenAI-compatible tool-call message order:

Graph 必须保持 OpenAI-compatible tool-call message 顺序：

```text
AIMessage(tool_calls=[call_1, call_2])
ToolMessage(tool_call_id=call_1.id)
ToolMessage(tool_call_id=call_2.id)
```

`call_model` validates this before hitting the provider. If the local history is invalid, the graph returns `transition_reason = "message_protocol_error"` instead of sending a request that would fail with a provider-side 400.

`call_model` 会在请求 provider 之前校验该协议。如果本地 history 不合法，graph 会返回 `transition_reason = "message_protocol_error"`，而不是发送会触发 provider-side 400 的请求。

`append_tool_results` is the only graph node that writes tool results back into `messages`. It converts each `ToolResultEnvelope` into a `ToolMessage`.

`append_tool_results` 是唯一把 tool result 写回 `messages` 的 graph node。它会把每个 `ToolResultEnvelope` 转为 `ToolMessage`。

## Tool Result Ordering / 工具结果排序

`execute_tools` merges newly executed results with pre-existing denied results and sorts them according to the latest assistant tool-call order. This keeps tool results stable even when some calls are denied and others are executed.

`execute_tools` 会把新执行结果和已有 denied results 合并，并按最新 assistant tool-call 顺序排序。即使部分调用被拒绝、部分调用被执行，tool result 顺序也保持稳定。

`ToolExecutionRuntime` preserves execution semantics:

`ToolExecutionRuntime` 保持以下执行语义：

- Consecutive read-only calls may run concurrently. / 连续只读调用可以并发。
- A write/bash/worktree call flushes pending reads before it executes. / write/bash/worktree 调用执行前会先 flush pending reads。
- Returned results keep the original tool-call order. / 返回结果保持原始 tool-call 顺序。

## CLI Streaming Behavior / CLI Streaming 行为

The CLI uses `graph.stream(..., stream_mode="updates")`.

CLI 使用 `graph.stream(..., stream_mode="updates")`。

For normal user turns:

普通用户回合：

```python
run_agent(user_input, state, thread_id)
```

For interrupted turns:

被 interrupt 的回合：

```python
resume_graph(resume_value, thread_id, state=state)
```

The runner synchronizes streamed state updates back into the caller-held `state` with `_sync_state_update`. This is required because the CLI keeps a local state object across turns while LangGraph also stores checkpointed state.

Runner 使用 `_sync_state_update` 把 streamed state update 同步回调用方持有的 `state`。这是必要的，因为 CLI 会跨回合持有本地 state，同时 LangGraph 也保存 checkpointed state。

The CLI renders progress for long phases:

CLI 会为较长阶段显示进度：

- permission approved or denied / 权限批准或拒绝
- tool execution completed / 工具执行完成
- transient model retry / transient 模型重试

This avoids the apparent hang where a tool has finished but the graph is waiting on the next model call.

这可以避免“工具已完成但 graph 正在等待下一次模型调用”时看起来像卡住的问题。

## Turn Preparation / 回合准备

Each new sync or async turn calls `_prepare_turn_state` before streaming:

每个新的同步或异步回合都会在 streaming 前调用 `_prepare_turn_state`：

```text
final_response = None
error = None
pending_question = False
pending_confirmation = False
pending_tool_calls = []
approved_tool_calls = []
tool_calls = []
tool_results = []
messages += HumanMessage(user_input)
```

This keeps `run_agent` and `run_agent_async` behavior aligned and prevents stale per-turn state from leaking into the next request.

这保持 `run_agent` 和 `run_agent_async` 行为一致，并防止上一回合的临时状态泄漏到下一次请求。

## Design Invariants / 设计不变量

- The graph runtime is LangGraph's `StateGraph`; this project does not reimplement checkpointing, interrupt, or streaming runtime behavior. / Graph runtime 是 LangGraph 的 `StateGraph`；本项目不重复实现 checkpointing、interrupt 或 streaming runtime。
- Every user turn and resume path supplies `thread_id`. / 每个用户回合和 resume 路径都提供 `thread_id`。
- Human approval is implemented with LangGraph `interrupt()` and `Command(resume=...)`. / 人工审批通过 LangGraph `interrupt()` 和 `Command(resume=...)` 实现。
- Side-effecting tools are never authorized as a group by a single approval. / 有副作用工具不会被一次审批批量授权。
- Denied tools produce normal `ToolResultEnvelope` records. / 被拒绝工具会产生普通 `ToolResultEnvelope` 记录。
- Tool results are appended as `ToolMessage` objects before the next model call. / 下一次模型调用前，tool result 会作为 `ToolMessage` 追加。
- Transient model retries are budgeted to avoid infinite loops. / transient 模型重试有预算，避免无限循环。
- CLI-held state is synchronized from streamed graph updates. / CLI 持有的 state 会从 streamed graph updates 同步。

## Tests Covering This Graph / 图相关测试

Useful regression tests:

相关回归测试：

- `tests/test_agent_graph.py`
- `tests/test_claude_code_agent_requirements.py`
- `tests/test_graph_diagram_export.py`
- `tests/test_integration_multi_turn.py`
- `tests/test_main_cli.py`
- `tests/test_message_protocol.py`
- `tests/test_recovery.py`

These tests cover stream state sync, permission interrupts/resume, multiple side-effect approvals, tool result ordering, message protocol validation, transient retry, CLI progress rendering, and LangGraph-generated diagram synchronization.

这些测试覆盖 stream state 同步、权限 interrupt/resume、多副作用工具审批、tool result 排序、message protocol 校验、transient retry、CLI 进度显示，以及 LangGraph 生成图的同步校验。
