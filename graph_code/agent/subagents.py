"""One-shot subagent implementation using a LangGraph subgraph."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph, add_messages

from ..config import Config


class SubagentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    prompt: str
    summary: str
    status: str


def _subagent_work(state: SubagentState) -> dict:
    prompt = state.get("prompt", "")
    return {
        "messages": [HumanMessage(content=prompt)],
        "summary": f"Subagent summary for: {prompt}",
        "status": "completed",
    }


def build_subagent_graph():
    graph = StateGraph(SubagentState)
    graph.add_node("work", _subagent_work)
    graph.add_edge(START, "work")
    graph.add_edge("work", END)
    return graph.compile()


def run_subagent(prompt: str, config: Config | None = None) -> dict:
    """Run a restricted one-shot worker and return a compact summary."""
    graph = build_subagent_graph()
    result = graph.invoke(
        {"messages": [], "prompt": prompt, "summary": "", "status": "pending"},
        {"configurable": {"thread_id": f"subagent-{abs(hash(prompt))}"}},
    )
    return {
        "status": result["status"],
        "summary": result["summary"],
        "thread_model": config.llm_model if config else "mock",
    }
