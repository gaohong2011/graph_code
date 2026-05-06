# Claude Code-like Context, Memory, and Prompt Design

Date: 2026-05-06
Status: approved design, awaiting written-spec review

## Context

Graph Code already has a LangGraph-based agent loop, a tool pipeline, and a strong first version of context compaction. Recent commits added protocol-safe `context_messages`, micro compact, summary compact, manual compact, reactive context-too-long retry, transcript writing, compact hooks, model-assisted compact summaries, and runtime rehydration.

The goal is to implement Claude Code-like behavior in this project while reusing those existing capabilities. The design intentionally does not replace the LangGraph graph. It adds new prompt, memory, session-memory, and rehydration modules around the existing `build_prompt -> call_model -> tool pipeline -> compact_check` flow.

The Claude Code source under `/Users/gaohong/go/src/claude-code` shows several transferable patterns:

- `src/constants/prompts.ts` and `src/constants/systemPromptSections.ts`: system prompt sections are separated into static and dynamic parts, with cache invalidation on compact or clear.
- `src/utils/claudemd.ts`: user, project, local, managed, rules, and auto-memory files are loaded in a deterministic priority order with include handling and truncation.
- `src/memdir/*`: long-term memory is file based, indexed by `MEMORY.md`, typed by frontmatter, and recalled selectively.
- `src/services/SessionMemory/*`: session memory is a markdown summary file maintained by a side process and used by compact.
- `src/services/compact/*`: compact prefers cheaper preserved context when available, protects tool-use/tool-result protocol pairs, writes compact boundaries, restores important context after compact, and treats failures as recoverable.

## Approved Direction

The chosen approach is a complete Claude Code-like architecture with layered implementation.

Accepted choices:

- Build the complete design now, but land implementation in internal layers rather than as one large rewrite.
- Background LLM features are implemented but default off.
- Long-term memory uses a global per-project path, not `.agent/memory/`, as the primary store.
- Memory is maintained through file-writing instructions rather than a large memory tool API.
- The system prompt should closely imitate Claude Code behavior, adapted for Graph Code.

## Goals

- Replace the hardcoded `SYSTEM_PROMPT` with a sectioned, cache-aware prompt builder.
- Load project instructions in a Claude Code-like way from `CLAUDE.md`, `.claude/CLAUDE.md`, and `.claude/rules/*.md`.
- Add a global file-based memory system under `~/.graph-code/projects/<project-key>/memory/`.
- Add optional session memory under `~/.graph-code/projects/<project-key>/session-memory/session.md`.
- Add optional background memory extraction and memory relevance selection, both disabled by default.
- Make compact prefer session memory when available and restore important runtime context after compaction.
- Preserve existing compaction protocol safety and current tests.

## Non-goals

- Do not port Claude Code's full TypeScript runtime, cache-editing APIs, telemetry, UI, or product-specific feature gates.
- Do not make background model calls by default.
- Do not let memory access become a general home-directory read/write escape hatch.
- Do not replace LangGraph checkpointing or the existing tool execution pipeline.
- Do not require provider-specific tokenization for the first implementation.

## Architecture

New modules:

```text
graph_code/agent/prompt/
  builder.py
  sections.py
  project_instructions.py
  cache.py

graph_code/agent/memory/
  paths.py
  types.py
  scan.py
  prompt.py
  relevance.py
  legacy.py

graph_code/agent/session_memory/
  state.py
  prompt.py
  updater.py
  compact.py

graph_code/agent/compaction/
  existing files plus enhanced runtime_context.py integration
```

State additions:

- `prompt_state`: section cache metadata, invalidation flags, and last prompt error.
- `memory_state`: surfaced memory paths, recent memory reads/writes, extraction cursor, and last memory error.
- `session_memory_state`: initialization flag, last summarized cursor/hash, tokens at last update, in-progress flag, and last error.
- `file_context_state`: recent file/tool context used for post-compact restoration.

The graph keeps its existing high-level shape. `build_prompt` becomes responsible for both context management and prompt construction. `call_model` receives the built system prompt instead of prepending the current hardcoded constant.

## System Prompt

`graph_code/agent/nodes.py` should stop owning the prompt text directly. A new `build_system_prompt(state, config)` builds the prompt from sections. The first implementation can still emit a single `SystemMessage` containing joined sections so message protocol behavior stays simple.

Sections:

- Identity: Graph Code is a Claude Code-like coding agent built on LangGraph.
- Task behavior: read code before editing, act on software-engineering context, avoid unrelated refactors, verify before completion, report failures honestly.
- Tool behavior: prefer dedicated tools for read/write/search, respect permission denials, avoid repeating denied calls, use parallel independent reads when possible.
- Context behavior: explain automatic compaction and old tool-result clearing; instruct the model to preserve important facts in replies, plans, or memory.
- Memory behavior: describe the file-based memory system, memory types, save/update/forget rules, and what not to store.
- Project instructions: inject loaded instruction files.
- Environment: working directory, git status summary if available, current date, model, permission mode, and memory paths.

Caching:

- Static and session-stable sections are cached in `prompt_state` or a small process-local section cache.
- Dynamic sections are rebuilt every turn.
- Compact, memory writes, instruction reloads, and worktree changes invalidate relevant cached sections.
- Prompt construction failures fall back to a minimal system prompt and record `prompt_state.last_error`.

Project instruction loading:

- Load in reverse priority order so later text has higher priority.
- Include managed and user instructions if configured later, but first implementation focuses on project-local files.
- Traverse from repository root or configured working root down to current working directory.
- Load `CLAUDE.md`, `.claude/CLAUDE.md`, and `.claude/rules/*.md`.
- Support frontmatter `paths` rules only when matching the active file context is available.
- Support relative includes only inside workspace or memory directory. Includes must not escape those roots.
- Strip frontmatter before injection and cap large files.

## Global Memory

Primary location:

```text
~/.graph-code/projects/<project-key>/memory/
```

`<project-key>` is derived from the canonical git root when available, otherwise from `WORKING_DIR`. It is converted to a stable safe slug.

Configuration:

- `GRAPH_CODE_MEMORY_DIR`: override the memory directory.
- `GRAPH_CODE_DISABLE_MEMORY=true`: disable global memory.
- `ENABLE_MEMORY_RELEVANCE=true`: enable model-assisted relevance selection.

Directory shape:

```text
memory/
  MEMORY.md
  user_preferences.md
  feedback_testing.md
  project_context.md
  references.md
```

Topic files use markdown with frontmatter:

```yaml
---
name: testing policy
description: Integration tests should use the real database
type: feedback
updated_at: 2026-05-06
---

The actual memory body.
```

Memory types:

- `user`: durable facts about the user's role, goals, knowledge, or preferences.
- `feedback`: user corrections or validated collaboration preferences.
- `project`: non-obvious project context that is not derivable from current files or git.
- `reference`: pointers to external systems, dashboards, documents, issue trackers, or channels.

What not to save:

- Code structure, conventions, or architecture that can be read from the repo.
- Git history or recent file changes.
- Secrets, credentials, API keys, tokens, or personal sensitive data.
- Temporary task state that only matters inside the current conversation.
- Anything already documented in project instruction files.

Tool access:

- The model maintains memory mostly through existing `read_file`, `write_file`, `edit_file`, and `search_files`.
- `ToolExecutionRuntime._safe_path` must explicitly allow both the workspace root and the configured memory root.
- Memory access must be limited to the configured memory directory, not all of the user's home directory.
- Existing `save_memory(namespace, key, value)` remains as a legacy shim. It writes or suggests a topic file in the global memory system, but the prompt should steer the model toward direct file maintenance.

Recall:

- By default, only `MEMORY.md` is injected, with line and byte caps.
- If `ENABLE_MEMORY_RELEVANCE=true`, scan topic frontmatter, ask a no-tools selector to choose up to five clearly relevant files, then add those files as context attachments.
- If relevance selection fails, inject only `MEMORY.md`.
- The model can still read topic files explicitly when the index suggests relevance.

## Session Memory

Session memory is optional and default off.

Configuration:

```bash
ENABLE_SESSION_MEMORY=true
SESSION_MEMORY_INIT_TOKENS=10000
SESSION_MEMORY_UPDATE_TOKENS=5000
SESSION_MEMORY_TOOL_CALLS=3
```

Path:

```text
~/.graph-code/projects/<project-key>/session-memory/session.md
```

Template sections:

- Session Title
- Current State
- Task Specification
- Files and Functions
- Workflow
- Errors & Corrections
- Codebase and System Documentation
- Learnings
- Key Results
- Worklog

Update trigger:

- Run only after a final response and only when no tool calls or permissions are pending.
- Initialize after estimated context tokens reach `SESSION_MEMORY_INIT_TOKENS`.
- Update after estimated growth since last extraction reaches `SESSION_MEMORY_UPDATE_TOKENS`.
- Also update when tool calls since last update reach `SESSION_MEMORY_TOOL_CALLS`, but only if the token threshold is also satisfied.

Execution model:

- First implementation uses a turn-end best-effort hook, not a separate always-running worker.
- The updater uses a no-tools model call to produce the complete new session memory content, then Graph Code writes the file locally.
- This avoids exposing write tools to the updater and avoids permission prompts.
- Failure records `session_memory_state.last_error` and does not affect the user-facing final response.

Compact integration:

- If enabled and `session.md` exists with non-template content, compact prefers it over a fresh model summary.
- The existing recent protocol-safe suffix is still preserved.
- If session memory is too large, truncate by section with a total budget.
- If session-memory compact would still exceed the compact threshold, fall back to existing summary compact.

## Auto Memory Extraction

Auto extraction is optional and default off.

Configuration:

```bash
ENABLE_AUTO_MEMORY_EXTRACTION=true
```

Behavior:

- Runs after final responses when the main model did not already write memory.
- Only saves memories when the user explicitly asks to remember or forget something, or when the conversation contains clear durable feedback.
- Uses the same memory type taxonomy and frontmatter.
- It should prefer updating existing topic files over creating duplicates.
- It must not inspect source files to create memory facts; extraction is based only on recent conversation content.

The first implementation can be conservative: support explicit "remember" and "forget" cases and simple durable feedback extraction. Broader heuristic extraction can be added later.

## Context Compaction Enhancements

Existing compaction remains the core:

- `AgentState.messages` keeps the full transcript.
- `context_messages` is the model-visible compacted context.
- Micro compact replaces old large compactable tool results.
- Summary compact creates a boundary and retains recent protocol-safe messages.
- Reactive retry handles provider context-too-long errors.

Enhancements:

- Pre-compact cleanup strips or replaces oversized non-text blocks where present.
- Session memory is used as the preferred summary source when enabled and valid.
- Post-compact rehydration inserts a runtime context message before the recent trailing tool group.
- Rehydration includes transcript path, session memory path, `MEMORY.md` index, relevant memory file list, recent key files, loaded skills, planning state, worktree state, MCP state, current task, and notifications.
- Recent key file restoration uses `file_context_state` gathered from read/search/edit/write tool calls.
- Do not reattach a file when the preserved suffix already contains its current read result.
- Compact completion invalidates prompt section caches and memory scan caches.
- Any post-compact attachment failure is recorded but does not interrupt the main loop.

Protocol safety:

- Continue using `split_recent_protocol_suffix`.
- Add tests for suffixes that include `ToolMessage` so the matching assistant `tool_calls` are always retained.
- Validate `[SystemMessage] + context_messages` with `validate_tool_message_protocol` before model calls.

## Data Flow

Per user turn:

1. `run_agent` appends the user message.
2. `build_prompt` runs context management and prompt construction.
3. Prompt construction loads cached sections, project instructions, memory index, optional relevant memories, and environment.
4. `call_model` sends the built system prompt plus `context_messages`.
5. Tool execution records envelopes and file access metadata.
6. Tool results are appended and compact checks run as they do today.
7. When a final response is produced, optional turn-end hooks update session memory and long-term memory.
8. Later compact operations use session memory, memory index, and file context for rehydration.

## Error Handling

- Prompt build failure: use a minimal system prompt and record the error.
- Instruction file failure: skip that file and continue.
- Memory directory failure: mark memory unavailable in prompt state and return clear tool errors for memory paths.
- Memory relevance failure: inject only `MEMORY.md`.
- Session memory update failure: record and continue.
- Auto memory extraction failure: record and continue.
- Session-memory compact failure: fall back to existing compact summary.
- Model summary failure: retain current extractive fallback and circuit breaker behavior.
- Rehydration failure: insert whatever context was available and record the missing parts.

## Security

- Workspace root and configured memory root are the only allowed filesystem roots.
- Memory root validation rejects empty, root, near-root, relative, UNC, and null-byte paths.
- Includes are limited to workspace and memory roots.
- Auto extraction must not save secrets or sensitive personal information.
- Background LLM features are disabled by default.
- Background writes are limited to memory/session-memory files owned by Graph Code.

## Testing

Unit tests:

- Prompt builder includes Claude Code-like behavior sections.
- Prompt cache invalidates after compact and memory changes.
- Project instruction loading respects priority and caps large files.
- Memory path slugging is stable and safe.
- Memory prompt includes the correct taxonomy and excludes disabled memory.
- Memory scanner parses frontmatter and `MEMORY.md` caps.
- Session memory update trigger respects token/tool thresholds.
- Session memory compact is preferred when valid.
- Session memory failures fall back to existing compact.
- File access tracking captures read/search/edit/write paths.
- Post-compact rehydration does not split tool protocol groups.

Integration tests:

- Long conversation compacts, writes transcript, preserves current request, and rehydrates memory/session/file context.
- Memory-enabled run can write a topic file outside workspace but inside the configured memory root.
- Memory-disabled run does not expose the memory root.
- Existing compaction tests continue to pass.
- Full `python -m pytest -q` remains green.

## Rollout Layers

This is one design but should be implemented in layers:

1. Prompt builder and project instruction loader.
2. Global memory paths, memory prompt, memory scan, and safe memory filesystem access.
3. File context tracking and compact rehydration enhancements.
4. Session memory updater and session-memory compact path, default off.
5. Optional memory relevance and auto memory extraction, default off.
6. Documentation and examples for enabling background features.

Each layer should preserve existing tests before moving to the next.
