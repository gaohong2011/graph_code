"""Node implementations for Graph Code LangGraph agent."""

import json
from typing import Any, Dict, List

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode

from ..config import get_config
from ..llm.client import get_llm
from ..tools.file_tools import read_file, write_file, list_directory, glob_search
from ..tools.code_tools import grep_search, read_code_chunk
from ..tools.exec_tools import bash_command, python_execute
from ..tools.interaction import ask_user, confirm_action, get_interaction_store
from .state import GraphCodeState


# Define system prompt
SYSTEM_PROMPT = """You are Graph Code, an AI programming assistant powered by LangGraph.
Your goal is to help users with coding tasks by reading, analyzing, and modifying files.

You have access to various tools for:
- File operations: read_file, write_file, list_directory, glob_search
- Code analysis: grep_search, read_code_chunk
- Execution: bash_command, python_execute
- Interaction: ask_user, confirm_action

Guidelines:
1. Always check if files exist before reading them
2. When modifying files, be precise and only change what's necessary
3. Use grep_search to find relevant code patterns
4. Execute commands to test your changes
5. Ask the user when you need clarification
6. Request confirmation for destructive operations (deleting files, major changes)

When writing code:
- Follow existing code style and conventions
- Add appropriate error handling
- Include comments for complex logic
- Test your code when possible

Be concise but thorough in your responses."""


def get_tools() -> List[StructuredTool]:
    """Get all available tools as StructuredTool instances."""
    from langchain_core.tools import tool as lc_tool

    @lc_tool
    def _read_file(file_path: str, offset: int = 0, limit: int = None):
        """Read content from a file."""
        return read_file(file_path, offset, limit)

    @lc_tool
    def _write_file(file_path: str, content: str, append: bool = False):
        """Write content to a file. Use with caution - can overwrite existing files."""
        return write_file(file_path, content, append)

    @lc_tool
    def _list_directory(dir_path: str = ".", recursive: bool = False):
        """List directory contents."""
        return list_directory(dir_path, recursive)

    @lc_tool
    def _glob_search(pattern: str, dir_path: str = "."):
        """Search for files matching a glob pattern."""
        return glob_search(pattern, dir_path)

    @lc_tool
    def _grep_search(pattern: str, path: str = ".", glob: str = None):
        """Search for pattern in files using regex."""
        return grep_search(pattern, path, glob)

    @lc_tool
    def _read_code_chunk(file_path: str, start_line: int, end_line: int = None, context_lines: int = 3):
        """Read a specific chunk of code with context."""
        return read_code_chunk(file_path, start_line, end_line, context_lines)

    @lc_tool
    def _bash_command(command: str, timeout: int = 60):
        """Execute a bash command. Be careful with destructive commands."""
        return bash_command(command, timeout)

    @lc_tool
    def _python_execute(code: str, timeout: int = 30):
        """Execute Python code."""
        return python_execute(code, timeout)

    @lc_tool
    def _ask_user(question: str):
        """Ask the user a question when you need clarification."""
        return ask_user(question)

    @lc_tool
    def _confirm_action(action: str, details: str = ""):
        """Request user confirmation for a sensitive action."""
        return confirm_action(action, details)

    return [
        _read_file, _write_file, _list_directory, _glob_search,
        _grep_search, _read_code_chunk,
        _bash_command, _python_execute,
        _ask_user, _confirm_action,
    ]


def agent_node(state: GraphCodeState) -> Dict[str, Any]:
    """Main agent node that decides what to do next."""
    config = get_config()
    llm = get_llm()

    # Bind tools to LLM
    tools = get_tools()
    llm_with_tools = llm.bind_tools(tools)

    # Prepare messages
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

    # Fix messages for kimi-k2.5 compatibility (add reasoning_content to tool call messages)
    _add_reasoning_content_to_messages(messages)

    # Check for pending interactions
    store = get_interaction_store()

    if store.pending_question:
        # We've asked a question, add it to messages
        messages.append(AIMessage(content=f"I need to ask: {store.pending_question}"))
        state["pending_question"] = True

    if store.pending_confirmation:
        # We're waiting for confirmation
        confirm = store.pending_confirmation
        messages.append(AIMessage(content=f"I need confirmation to: {confirm['action']}\n{confirm.get('details', '')}"))
        state["pending_confirmation"] = True

    # Call LLM
    response = llm_with_tools.invoke(messages)

    # Handle kimi-k2.5 reasoning_content requirement
    # The API expects reasoning_content in assistant messages with tool calls
    if hasattr(response, 'tool_calls') and response.tool_calls:
        # Add empty reasoning_content to satisfy API requirement
        if 'reasoning_content' not in response.additional_kwargs:
            response.additional_kwargs['reasoning_content'] = ''
        return {
            "messages": [response],
            "tool_calls": response.tool_calls,
        }

    # Regular response
    return {
        "messages": [response],
        "final_response": response.content,
    }


def _add_reasoning_content_to_messages(messages: list) -> list:
    """Add reasoning_content to AIMessages with tool_calls for kimi-k2.5 compatibility."""
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            if 'reasoning_content' not in msg.additional_kwargs:
                msg.additional_kwargs['reasoning_content'] = ''
    return messages


def tools_node(state: GraphCodeState) -> Dict[str, Any]:
    """Execute tool calls."""
    tool_calls = state.get("tool_calls", [])
    if not tool_calls:
        return {}

    results = []
    tool_node = ToolNode(get_tools())

    # Execute tools
    for tool_call in tool_calls:
        # Safely extract tool_call_id (handle missing or empty id)
        tool_call_id = tool_call.get("id") or "unknown"

        try:
            result = tool_node.invoke({
                "messages": [AIMessage(content="", tool_calls=[tool_call])]
            })
            if result and "messages" in result:
                # Verify tool results have valid tool_call_ids
                for msg in result["messages"]:
                    if isinstance(msg, ToolMessage) and not msg.tool_call_id:
                        msg.tool_call_id = tool_call_id
                results.extend(result["messages"])
        except Exception as e:
            results.append(ToolMessage(
                content=f"Error: {str(e)}",
                tool_call_id=tool_call_id
            ))

    # Fix messages in state for kimi-k2.5 compatibility
    _add_reasoning_content_to_messages(state.get("messages", []))

    # Clear tool calls after execution
    return {
        "tool_calls": [],
        "tool_results": results,
        "messages": results,
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def check_interaction_node(state: GraphCodeState) -> Dict[str, Any]:
    """Check if we need to pause for user interaction."""
    store = get_interaction_store()

    if store.pending_question or store.pending_confirmation:
        return {
            "pending_question": bool(store.pending_question),
            "pending_confirmation": bool(store.pending_confirmation),
        }

    return {}


def handle_interaction_response(state: GraphCodeState, user_input: str) -> Dict[str, Any]:
    """Handle user response to pending interaction."""
    store = get_interaction_store()

    # Add user response to messages
    messages = [HumanMessage(content=user_input)]

    # Clear pending state
    store.clear()

    return {
        "messages": messages,
        "pending_question": False,
        "pending_confirmation": False,
        "interaction_result": user_input,
    }


def should_continue(state: GraphCodeState) -> str:
    """Decide whether to continue execution or end."""
    # Check for pending interactions
    if state.get("pending_question") or state.get("pending_confirmation"):
        return "pause"

    # Check if we have a final response
    if state.get("final_response"):
        return "end"

    # Check for errors
    if state.get("error"):
        return "end"

    # Check iteration limit
    config = get_config()
    if state.get("iteration_count", 0) >= config.max_tool_iterations:
        return "end"

    # Check if there are pending tool calls
    if state.get("tool_calls"):
        return "execute_tools"

    # Default: end
    return "end"
