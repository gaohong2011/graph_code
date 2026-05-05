"""Export the compiled LangGraph topology for documentation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod

from graph_code.agent.graph import build_agent
from graph_code.config import Config


@dataclass(frozen=True)
class DiagramExport:
    """Paths written by the graph diagram exporter."""

    mermaid_path: Path
    png_path: Path | None
    mermaid: str


def export_stategraph_diagram(
    output_dir: str | Path = "docs/assets",
    *,
    write_png: bool = False,
    draw_method: MermaidDrawMethod = MermaidDrawMethod.API,
    curve_style: CurveStyle = CurveStyle.LINEAR,
    background_color: str = "white",
    padding: int = 10,
) -> DiagramExport:
    """Export Mermaid source and, optionally, a PNG from the compiled graph.

    The compiled LangGraph graph is the only topology source here. This keeps
    the documentation diagram aligned with the actual StateGraph edges.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    graph = build_agent(config=Config.for_tests(Path.cwd(), model="mock")).get_graph()
    mermaid = graph.draw_mermaid(curve_style=curve_style, wrap_label_n_words=3)

    mermaid_path = output_path / "stategraph-topology.mmd"
    mermaid_path.write_text(mermaid, encoding="utf-8")

    png_path = None
    if write_png:
        png_path = output_path / "stategraph-topology.png"
        graph.draw_mermaid_png(
            curve_style=curve_style,
            output_file_path=str(png_path),
            draw_method=draw_method,
            background_color=background_color,
            padding=padding,
            wrap_label_n_words=3,
        )

    return DiagramExport(mermaid_path=mermaid_path, png_path=png_path, mermaid=mermaid)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export docs/assets/stategraph-topology from the compiled LangGraph graph."
    )
    parser.add_argument("--output-dir", default="docs/assets", help="Directory for generated files.")
    parser.add_argument("--png", action="store_true", help="Also render a PNG.")
    parser.add_argument(
        "--draw-method",
        choices=[method.value for method in MermaidDrawMethod],
        default=MermaidDrawMethod.API.value,
        help="LangGraph Mermaid PNG renderer to use.",
    )
    parser.add_argument(
        "--curve-style",
        choices=[style.value for style in CurveStyle],
        default=CurveStyle.LINEAR.value,
        help="Mermaid curve style passed through LangGraph.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = export_stategraph_diagram(
        args.output_dir,
        write_png=args.png,
        draw_method=MermaidDrawMethod(args.draw_method),
        curve_style=CurveStyle(args.curve_style),
    )
    print(f"Wrote {result.mermaid_path}")
    if result.png_path:
        print(f"Wrote {result.png_path}")


if __name__ == "__main__":
    main()
