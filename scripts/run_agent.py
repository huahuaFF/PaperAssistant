"""Validate that the complete Agent graph topology compiles.

This command intentionally does not invoke the graph: nodes are implemented in
later steps, and placeholders raise an error if executed.
"""

from app.graph.nodes.contracts import create_placeholder_nodes
from app.graph.research_graph import build_research_graph


def main() -> None:
    graph = build_research_graph(create_placeholder_nodes())
    print(graph.get_graph().draw_mermaid())


if __name__ == "__main__":
    main()
