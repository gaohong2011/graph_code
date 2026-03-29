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

from .agent.graph import build_agent, resume_with_interaction, run_agent
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
    if args.working_dir:
        config.working_dir = args.working_dir
    if args.auto_confirm:
        config.auto_confirm = True

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

                # Update state
                for key, value in event.items():
                    if key in state:
                        if key == "messages":
                            # Merge messages carefully
                            existing_content = {m.content for m in state["messages"]}
                            for msg in value:
                                if msg.content not in existing_content:
                                    state["messages"].append(msg)
                        elif key == "tool_results":
                            state["tool_results"].extend(value)
                        else:
                            state[key] = value

                # Check for final response
                if event.get("final_response"):
                    console.print(f"\n[bold green]Graph Code:[/bold green]")
                    format_message(event["final_response"], console)
                    console.print()

                # Check for pending interaction
                if event.get("pending_question") or event.get("pending_confirmation"):
                    interaction_result = handle_pending_interaction(state, console)
                    if interaction_result:
                        # Resume with interaction response
                        for resume_event in resume_with_interaction(state, interaction_result, thread_id):
                            for key, value in resume_event.items():
                                if key in state:
                                    if key == "messages":
                                        existing_content = {m.content for m in state["messages"]}
                                        for msg in value:
                                            if msg.content not in existing_content:
                                                state["messages"].append(msg)
                                    else:
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
        for event in run_agent(user_input, state, thread_id):
            # Update state
            for key, value in event.items():
                if key in state:
                    state[key] = value

            # Check for final response
            if event.get("final_response"):
                format_message(event["final_response"], console)

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
        "--yes",
        action="store_true",
        help="Answer yes to all confirmations (for single command mode)"
    )

    args = parser.parse_args()

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
