"""Debug utilities for Graph Code.

Provides logging and tracing for LLM interactions and tool execution.
"""

import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from ..config import get_config


class DebugCallbackHandler(BaseCallbackHandler):
    """Callback handler that logs LLM interactions to console and/or file.

    Set DEBUG=true or DEBUG_LLM=true to enable.
    """

    def __init__(self, log_file: Optional[str] = None):
        """Initialize handler.

        Args:
            log_file: Optional file path to write logs to
        """
        self.log_file = log_file
        self._session_start = datetime.now()
        self._interaction_count = 0

    def _log(self, title: str, data: Any, is_request: bool = True):
        """Log a debug message."""
        config = get_config()
        if not (config.debug or config.debug_llm):
            return

        self._interaction_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        # Build output
        lines = []
        lines.append("")
        lines.append("=" * 80)
        direction = ">>> REQUEST" if is_request else "<<< RESPONSE"
        lines.append(f"[{timestamp}] {direction}: {title}")
        lines.append("=" * 80)

        # Format data
        if isinstance(data, dict):
            lines.append(json.dumps(data, indent=2, ensure_ascii=False, default=str))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                lines.append(f"[{i}] {self._format_item(item)}")
        else:
            lines.append(str(data))

        lines.append("=" * 80)
        output = "\n".join(lines)

        # Print to console
        print(output, file=sys.stderr)

        # Write to file if configured
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(output + "\n")

    def _format_item(self, item: Any) -> str:
        """Format a single item for logging."""
        if hasattr(item, "content"):
            content = item.content
            if hasattr(item, "tool_calls") and item.tool_calls:
                return f"{type(item).__name__}: {content[:100]}... [tool_calls: {len(item.tool_calls)}]"
            if hasattr(item, "tool_call_id"):
                return f"{type(item).__name__}: {content[:100]}... [tool_call_id: {item.tool_call_id}]"
            return f"{type(item).__name__}: {content[:100]}..."
        return str(item)[:200]

    # LLM callbacks
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> None:
        """Log when LLM starts processing."""
        self._log(
            "LLM Start",
            {
                "model": serialized.get("repr", "unknown"),
                "prompt_count": len(prompts),
                "prompts": prompts[:3] if prompts else [],  # Limit to first 3
            },
            is_request=True,
        )

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Log when LLM completes."""
        generations = response.generations[0] if response.generations else []
        outputs = []
        for gen in generations:
            if hasattr(gen, "message"):
                msg = gen.message
                out = {
                    "type": type(msg).__name__,
                    "content": msg.content[:500] if msg.content else None,
                }
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    out["tool_calls"] = msg.tool_calls
                if hasattr(msg, "additional_kwargs") and msg.additional_kwargs:
                    out["additional_kwargs"] = msg.additional_kwargs
                outputs.append(out)
            else:
                outputs.append({"text": gen.text[:500] if gen.text else None})

        self._log(
            "LLM End",
            {
                "output_count": len(outputs),
                "outputs": outputs,
            },
            is_request=False,
        )

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """Log LLM errors."""
        self._log(
            "LLM ERROR",
            {
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
            is_request=False,
        )

    # Chat model callbacks
    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        **kwargs: Any,
    ) -> None:
        """Log when chat model starts."""
        # Flatten messages for logging
        all_messages = []
        for msg_list in messages:
            for msg in msg_list:
                entry = {
                    "role": type(msg).__name__,
                    "content": msg.content[:500] if msg.content else None,
                }
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": tc.get("id", "N/A"),
                            "name": tc.get("name", "N/A"),
                            "args": tc.get("args", {}),
                        }
                        for tc in msg.tool_calls[:5]  # Limit to 5
                    ]
                if hasattr(msg, "tool_call_id"):
                    entry["tool_call_id"] = msg.tool_call_id
                all_messages.append(entry)

        self._log(
            "Chat Model Start",
            {
                "model": serialized.get("repr", "unknown"),
                "message_count": len(all_messages),
                "messages": all_messages,
            },
            is_request=True,
        )


def get_debug_callbacks() -> List[BaseCallbackHandler]:
    """Get list of debug callbacks based on configuration."""
    config = get_config()
    callbacks = []

    if config.debug or config.debug_llm:
        # Check for log file
        log_file = os.getenv("DEBUG_LOG_FILE")
        callbacks.append(DebugCallbackHandler(log_file=log_file))

    return callbacks


def log_tool_execution(tool_name: str, args: Dict[str, Any], result: Any):
    """Log a tool execution.

    Args:
        tool_name: Name of the tool
        args: Tool arguments
        result: Tool result
    """
    config = get_config()
    if not (config.debug or config.debug_llm):
        return

    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

    lines = []
    lines.append("")
    lines.append("-" * 80)
    lines.append(f"[{timestamp}] TOOL EXECUTION: {tool_name}")
    lines.append("-" * 80)
    lines.append(f"Arguments: {json.dumps(args, indent=2, ensure_ascii=False, default=str)}")
    lines.append(f"Result: {str(result)[:1000]}")
    lines.append("-" * 80)

    output = "\n".join(lines)
    print(output, file=sys.stderr)

    # Also write to file if configured
    log_file = os.getenv("DEBUG_LOG_FILE")
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(output + "\n")


def log_state_transition(node_name: str, state: Dict[str, Any]):
    """Log a state transition in the graph.

    Args:
        node_name: Name of the node being executed
        state: Current state (will be summarized)
    """
    config = get_config()
    if not config.debug:
        return

    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

    # Summarize state
    summary = {
        "messages_count": len(state.get("messages", [])),
        "tool_calls_count": len(state.get("tool_calls", [])),
        "tool_results_count": len(state.get("tool_results", [])),
        "iteration": state.get("iteration_count", 0),
        "has_final_response": state.get("final_response") is not None,
        "has_error": state.get("error") is not None,
    }

    lines = []
    lines.append("")
    lines.append(f"[{timestamp}] STATE -> {node_name}")
    lines.append(f"  {json.dumps(summary, indent=2)}")

    output = "\n".join(lines)
    print(output, file=sys.stderr)
