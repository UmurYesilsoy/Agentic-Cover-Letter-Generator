"""Thin loader that turns cover_letter_v2.ipynb into an importable `graph`.

The notebook is the source of truth - prompts, nodes, the graph itself all live there and get
edited there. This module exists only because a plain Python host (LangGraph Studio, `langgraph
dev`) needs something importable to point at; it works by reading the notebook's own code cells
and executing them in order, the same trick run_headless.py uses to smoke-test the notebook from
the terminal.

A line tagged `# SHIM:SKIP` is notebook-only (eagerly loading inputs/ for interactive use) and is
dropped before execution - Studio invokes the graph itself instead, via the `load` node.
"""
import json
from pathlib import Path

NOTEBOOK = Path(__file__).parent / "cover_letter_v2.ipynb"


def _build_graph():
    cells = [
        "".join(cell["source"])
        for cell in json.loads(NOTEBOOK.read_text())["cells"]
        if cell["cell_type"] == "code"
    ]
    compile_at = next(i for i, source in enumerate(cells) if "builder.compile(" in source)

    namespace: dict = {"__name__": "__main__"}
    for index, source in enumerate(cells[: compile_at + 1]):
        filtered = "\n".join(line for line in source.splitlines() if "SHIM:SKIP" not in line)
        exec(compile(filtered, f"<{NOTEBOOK.name} cell {index}>", "exec"), namespace)

    return namespace["graph"]


graph = _build_graph()
