"""Tests for LangGraph-generated topology documentation."""

from __future__ import annotations

from graph_code.utils.export_graph_diagram import export_stategraph_diagram


def test_stategraph_diagram_export_uses_compiled_langgraph(tmp_path):
    result = export_stategraph_diagram(tmp_path)
    mermaid = result.mermaid_path.read_text(encoding="utf-8")

    assert result.png_path is None
    assert result.mermaid == mermaid
    assert "graph TD;" in mermaid
    assert "__start__ --> drain_notifications" in mermaid
    assert "permission_gate -. &nbsp;interrupt&nbsp; .-> human_permission_interrupt" in mermaid
    assert "recovery_handler_after_tools --> call_model" in mermaid
    assert "final_response --> __end__" in mermaid
