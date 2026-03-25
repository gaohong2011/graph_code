"""LangGraph builder and runner for Graph Code."""

from typing import AsyncIterator, Iterator, Optional

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

from ..config import get_config
from .nodes import (
    agent_node,
    tools_node,
    check_interaction_node,
    handle_interaction_response,
    should_continue,
)
from .state import GraphCodeState, create_initial_state


def build_agent() -> StateGraph:
    """Build the Graph Code agent as a LangGraph.

    Returns:
        Compiled StateGraph ready for execution.
    """
    # Create graph
    workflow = StateGraph(GraphCodeState)

    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)
    workflow.add_node("check_interaction", check_interaction_node)

    # Set entry point
    workflow.set_entry_point("agent")

    # Add conditional edges from agent
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "execute_tools": "tools",
            "pause": "check_interaction",
            "end": END,
        }
    )

    # Add edge from tools back to agent
    workflow.add_edge("tools", "agent")

    # Add edge from check_interaction to END (pause for user)
    workflow.add_edge("check_interaction", END)

    # Compile the graph
    return workflow.compile()


def run_agent(
    user_input: str,
    state: Optional[GraphCodeState] = None,
    thread_id: Optional[str] = None,
) -> Iterator[GraphCodeState]:
    """Run the agent with user input.

    Args:
        user_input: The user's message
        state: Optional existing state to continue from
        thread_id: Optional thread ID for persistence

    Yields:
        State snapshots during execution
    """
    # Build agent
    agent = build_agent()

    # Initialize or use existing state
    if state is None:
        state = create_initial_state()

    # Add user message
    state["messages"].append(HumanMessage(content=user_input))

    # Run the agent
    config = {"configurable": {"thread_id": thread_id or "default"}}

    for event in agent.stream(state, config):
        if isinstance(event, dict):
            # LangGraph returns events as {node_name: state_update}
            # Extract the actual state update from the node output
            for node_name, node_output in event.items():
                if isinstance(node_output, dict):
                    yield node_output


async def run_agent_async(
    user_input: str,
    state: Optional[GraphCodeState] = None,
    thread_id: Optional[str] = None,
) -> AsyncIterator[GraphCodeState]:
    """Run the agent asynchronously.

    Args:
        user_input: The user's message
        state: Optional existing state to continue from
        thread_id: Optional thread ID for persistence

    Yields:
        State snapshots during execution
    """
    # Build agent
    agent = build_agent()

    # Initialize or use existing state
    if state is None:
        state = create_initial_state()

    # Add user message
    state["messages"].append(HumanMessage(content=user_input))

    # Run the agent
    config = {"configurable": {"thread_id": thread_id or "default"}}

    async for event in agent.astream(state, config):
        if isinstance(event, dict):
            yield event


def resume_with_interaction(
    state: GraphCodeState,
    user_response: str,
    thread_id: Optional[str] = None,
) -> Iterator[GraphCodeState]:
    """Resume agent execution after user interaction.

    Args:
        state: The current state (paused at interaction)
        user_response: The user's response
        thread_id: Optional thread ID

    Yields:
        State snapshots during execution
    """
    # Update state with user response
    updates = handle_interaction_response(state, user_response)
    for key, value in updates.items():
        if key == "messages":
            state["messages"].extend(value)
        else:
            state[key] = value

    # Build and run agent
    agent = build_agent()
    config = {"configurable": {"thread_id": thread_id or "default"}}

    for event in agent.stream(state, config):
        if isinstance(event, dict):
            yield event
