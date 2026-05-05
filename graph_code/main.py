"""CLI entry point for Graph Code."""

import argparse
import os
import sys
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax

from .agent.graph import resume_graph, resume_with_interaction, run_agent
from .agent.state import create_initial_state
from .config import get_config
from .tools.interaction import get_interaction_store


def print_banner(console: Console):
    """Print welcome banner."""
    banner = r"""
   _____                 _       _____          _
  / ____|               | |     / ____|        | |
 | |  __ _ __ __ _ _ __ | |__  | |     ___   __| | ___
 | | |_ | '__/ _` | '_ \| '_ \ | |    / _ \ / _` |/ _ \
 | |__| | | | (_| | |_) | | | || |___| (_) | (_| |  __/
  \_____|_|  \__,_| .__/|_| |_| \_____\___/ \__,_|\___|
                  | |
                  |_|
    """
    console.print(Panel(banner, title="AI Programming Assistant", border_style="blue"))


def setup_config(args) -> bool:
    """Setup and validate configuration.

    Returns:
        True if config is valid, False otherwise.
    """
    config = get_config()

    # Override with command line args
    if args.api_key:
        config.llm_api_key = args.api_key
    if args.base_url:
        config.llm_base_url = args.base_url
    if args.model:
        config.llm_model = args.model
    if args.mock:
        config.llm_model = "mock"
    if args.working_dir:
        config.working_dir = args.working_dir
    if args.auto_confirm:
        config.auto_confirm = True
    if args.permission_mode:
        config.permission_mode = args.permission_mode

    # Validate
    errors = config.validate()
    if errors:
        console = Console()
        console.print("[red]Configuration Error:[/red]")
        for error in errors:
            console.print(f"  - {error}")
        console.print("\n[yellow]Set environment variables:[/yellow]")
        console.print("  export LLM_API_KEY=your_api_key")
        console.print("  export LLM_BASE_URL=https://api.openai.com/v1  # optional")
        console.print("  export LLM_MODEL=gpt-4o-mini  # optional")
        return False

    return True


def handle_pending_interaction(state, console: Console) -> Optional[str]:
    """Handle pending user interaction.

    Returns:
        User response string, or None if no pending interaction.
    """
    store = get_interaction_store()

    if store.pending_question:
        console.print(f"\n[yellow]Question:[/yellow] {store.pending_question}")
        response = Prompt.ask("Your answer")
        return response

    if store.pending_confirmation:
        action = store.pending_confirmation["action"]
        details = store.pending_confirmation.get("details", "")

        console.print(f"\n[yellow]Confirmation Required:[/yellow]")
        console.print(f"Action: {action}")
        if details:
            console.print(f"Details: {details}")

        confirmed = Confirm.ask("Proceed?")
        return "yes" if confirmed else "no"

    return None


def handle_graph_interrupt(event: dict, console: Console, auto_yes: bool = False) -> dict:
    """Ask the user to approve or deny a LangGraph interrupt."""
    interrupts = event.get("__interrupt__") or []
    interrupt_value = interrupts[0].value if interrupts else {}
    tool_name = interrupt_value.get("tool_name", "unknown")
    reason = interrupt_value.get("reason", "Permission required")
    args = interrupt_value.get("args", {})

    console.print("\n[yellow]Permission Required:[/yellow]")
    console.print(f"Tool: {tool_name}")
    console.print(f"Reason: {reason}")
    if args:
        _print_interrupt_args(args, console)

    approved = auto_yes or Confirm.ask("Approve?")
    if approved:
        return {"approved": True}
    denial_reason = Prompt.ask("Reason", default="denied")
    return {"approved": False, "reason": denial_reason}


def format_message(content: str, console: Console):
    """Format and display message content."""
    # Try to detect and format code blocks
    if "```" in content:
        console.print(Markdown(content))
    else:
        console.print(content)


def run_interactive(console: Console, args):
    """Run in interactive REPL mode."""
    console.print("\n[green]Interactive mode. Type 'exit' or 'quit' to exit.[/green]\n")

    # Initialize state
    state = create_initial_state()
    thread_id = args.thread_id or "default"

    while True:
        try:
            # Get user input
            user_input = Prompt.ask("[bold blue]You[/bold blue]")

            if user_input.lower() in ("exit", "quit", "q"):
                console.print("[yellow]Goodbye![/yellow]")
                break

            if user_input.lower() in ("help", "?"):
                print_help(console)
                continue

            if not user_input.strip():
                continue

            # Run agent
            console.print("\n[dim]Thinking...[/dim]")
            event_count = 0

            for event in run_agent(user_input, state, thread_id):
                event_count += 1

                if isinstance(event, dict) and "__interrupt__" in event:
                    resume_value = handle_graph_interrupt(
                        event,
                        console,
                        auto_yes=args.yes or get_config().auto_confirm,
                    )
                    _resume_until_complete(resume_value, thread_id, state, console, args)
                    continue

                # Update state (skip messages - managed by LangGraph internally)
                for key, value in event.items():
                    if key in state and key != "messages":
                        if key == "tool_results":
                            state["tool_results"].extend(value)
                        else:
                            state[key] = value

                # Check for final response
                if event.get("final_response"):
                    console.print(f"\n[bold green]Graph Code:[/bold green]")
                    format_message(event["final_response"], console)
                    console.print()
                else:
                    _render_node_progress(event, console)

                # Check for pending interaction
                if event.get("pending_question") or event.get("pending_confirmation"):
                    interaction_result = handle_pending_interaction(state, console)
                    if interaction_result:
                        # Resume with interaction response
                        for resume_event in resume_with_interaction(state, interaction_result, thread_id):
                            for key, value in resume_event.items():
                                if key in state and key != "messages":
                                    state[key] = value

                            if resume_event.get("final_response"):
                                console.print(f"\n[bold green]Graph Code:[/bold green]")
                                format_message(resume_event["final_response"], console)
                                console.print()

            if event_count == 0:
                console.print("[yellow]No response generated.[/yellow]")

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")


def run_single_command(console: Console, args):
    """Run a single command and exit."""
    user_input = args.command
    state = create_initial_state()
    thread_id = args.thread_id or "single"

    console.print(f"[dim]Executing: {user_input}[/dim]\n")

    try:
        for event in run_agent(
            user_input,
            state,
            thread_id,
            stream_mode=args.stream_mode,
        ):
            if isinstance(event, dict) and "__interrupt__" in event:
                resume_value = handle_graph_interrupt(
                    event,
                    console,
                    auto_yes=args.yes or get_config().auto_confirm,
                )
                _resume_until_complete(resume_value, thread_id, state, console, args)
                continue

            if isinstance(event, tuple):
                mode, payload = event
                if mode == "messages":
                    console.print(payload)
                elif mode == "custom":
                    console.print(payload)
                elif mode == "updates":
                    for node_output in _iter_node_outputs(payload):
                        if node_output.get("final_response"):
                            format_message(node_output["final_response"], console)
                        else:
                            _render_node_progress(node_output, console)
                continue

            # Update state
            for key, value in event.items():
                if key in state:
                    state[key] = value

            # Check for final response
            if event.get("final_response"):
                format_message(event["final_response"], console)
            else:
                _render_node_progress(event, console)

            # Handle pending interaction
            if event.get("pending_question") or event.get("pending_confirmation"):
                if args.yes or get_config().auto_confirm:
                    interaction_result = "yes"
                else:
                    interaction_result = handle_pending_interaction(state, console)

                if interaction_result:
                    for resume_event in resume_with_interaction(state, interaction_result, thread_id):
                        if resume_event.get("final_response"):
                            format_message(resume_event["final_response"], console)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


def _iter_node_outputs(event):
    if isinstance(event, dict):
        if any(isinstance(value, dict) for value in event.values()):
            for value in event.values():
                if isinstance(value, dict):
                    yield value
            return
        if "final_response" in event or "tool_results" in event:
            yield event
            return


def _resume_until_complete(resume_value, thread_id, state, console: Console, args) -> list[str]:
    """Resume a graph interrupt, handling any nested interrupts in sequence."""
    finals: list[str] = []
    next_resume = resume_value
    while True:
        nested_resume = None
        for resume_event in resume_graph(next_resume, thread_id, state=state):
            if isinstance(resume_event, dict) and "__interrupt__" in resume_event:
                nested_resume = handle_graph_interrupt(
                    resume_event,
                    console,
                    auto_yes=args.yes or get_config().auto_confirm,
                )
                break
            for node_output in _iter_node_outputs(resume_event):
                if node_output.get("final_response"):
                    final = node_output["final_response"]
                    finals.append(final)
                    format_message(final, console)
                else:
                    _render_node_progress(node_output, console)
        if nested_resume is None:
            return finals
        next_resume = nested_resume


def _render_node_progress(node_output: dict, console: Console) -> None:
    """Render concise progress updates for long graph phases."""
    reason = node_output.get("transition_reason")
    if reason == "permission_approved":
        console.print("[dim]Permission approved. Resuming...[/dim]")
        return
    if reason == "permission_denied":
        console.print("[dim]Permission denied. Continuing...[/dim]")
        return
    if reason == "transient_model_retry":
        console.print("[dim]Model call was transiently unavailable. Retrying...[/dim]")
        return
    if reason == "tools_executed":
        names = _tool_names_from_results(node_output.get("tool_results") or [])
        label = ", ".join(names) if names else "tool"
        console.print(f"[dim]Completed tool: {label}. Waiting for model response...[/dim]")


def _tool_names_from_results(tool_results) -> list[str]:
    names: list[str] = []
    for result in tool_results:
        metadata = result.get("metadata", {}) if isinstance(result, dict) else getattr(result, "metadata", {})
        name = metadata.get("tool_name") if isinstance(metadata, dict) else None
        if name and name not in names:
            names.append(name)
    return names


def _preview_interrupt_args(args):
    if isinstance(args, dict):
        return {key: _preview_interrupt_args(value) for key, value in args.items()}
    if isinstance(args, list):
        preview = [_preview_interrupt_args(value) for value in args[:8]]
        if len(args) > 8:
            preview.append(f"... [truncated, {len(args)} items]")
        return preview
    if isinstance(args, str) and len(args) > 240:
        return f"{args[:80]}... [truncated, {len(args)} chars]"
    return args


def _print_interrupt_args(args, console: Console) -> None:
    console.print("Args:")
    preview = _preview_interrupt_args(args)
    if isinstance(preview, dict):
        for key, value in preview.items():
            console.print(f"  {key}: {value}", markup=False)
        return
    console.print(f"  {preview}", markup=False)


def print_help(console: Console):
    """Print help information."""
    help_text = """
## Graph Code Commands

- `exit`, `quit`, `q` - Exit the program
- `help`, `?` - Show this help message

## Environment Variables

- `LLM_API_KEY` - Your LLM API key (required)
- `LLM_BASE_URL` - Base URL for LLM API (optional)
- `LLM_MODEL` - Model name (default: gpt-4o-mini)
- `WORKING_DIR` - Working directory (default: current directory)
- `AUTO_CONFIRM` - Skip confirmations (default: false)

## Example Usage

```
graph-code
graph-code "List all Python files in the current directory"
graph-code --model gpt-4 "Refactor this function"
```
    """
    console.print(Markdown(help_text))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Graph Code - AI Programming Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  graph-code                          # Start interactive mode
  graph-code "list all python files"  # Run single command
  graph-code --model gpt-4            # Use specific model
  graph-code --working-dir /path      # Set working directory
        """
    )

    parser.add_argument(
        "command",
        nargs="?",
        help="Command to execute (if not provided, starts interactive mode)"
    )

    parser.add_argument(
        "--api-key",
        help="LLM API key (or set LLM_API_KEY env var)"
    )

    parser.add_argument(
        "--base-url",
        help="LLM base URL (or set LLM_BASE_URL env var)"
    )

    parser.add_argument(
        "--model",
        "-m",
        help="Model name (default: gpt-4o-mini)"
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use the built-in mock model (no API key required)"
    )

    parser.add_argument(
        "--working-dir",
        "-w",
        help="Working directory (default: current directory)"
    )

    parser.add_argument(
        "--thread-id",
        "-t",
        help="Thread ID for conversation persistence"
    )

    parser.add_argument(
        "--auto-confirm",
        "-y",
        action="store_true",
        help="Automatically confirm all actions (use with caution)"
    )

    parser.add_argument(
        "--permission-mode",
        choices=["default", "plan", "auto"],
        default=None,
        help="Tool permission mode"
    )

    parser.add_argument(
        "--stream-mode",
        default="updates",
        help="LangGraph stream mode: updates, messages, custom, or comma-separated modes"
    )

    parser.add_argument(
        "--yes",
        action="store_true",
        help="Answer yes to all confirmations (for single command mode)"
    )

    args = parser.parse_args()
    if "," in args.stream_mode:
        args.stream_mode = [item.strip() for item in args.stream_mode.split(",") if item.strip()]

    # Setup console
    console = Console()

    # Print banner
    print_banner(console)

    # Setup config
    if not setup_config(args):
        sys.exit(1)

    config = get_config()
    console.print(f"[dim]Model: {config.llm_model}[/dim]")
    console.print(f"[dim]Working dir: {config.working_path}[/dim]\n")

    # Run in appropriate mode
    if args.command:
        run_single_command(console, args)
    else:
        run_interactive(console, args)


if __name__ == "__main__":
    main()
