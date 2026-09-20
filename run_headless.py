"""Run the notebook from the terminal, without Jupyter.

    python run_headless.py --dry-run    # structure only, no API calls, free
    python run_headless.py              # full run, auto-approves at the review gate

Useful as a smoke test after changing a node. The notebook itself remains the real interface -
this just executes its code cells in order.
"""
import argparse
import json
import sys
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

NOTEBOOK = Path(__file__).parent / "cover_letter_agent.ipynb"


def locate(cells):
    """Find the diagram cell and the compile cell by content, so edits cannot desync them."""
    render = next(i for i, c in enumerate(cells) if "draw_mermaid_png" in c)
    compile_at = next(i for i, c in enumerate(cells) if "builder.compile(" in c)
    return render, compile_at


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="stop once the graph compiles - makes no API calls")
    args = parser.parse_args()

    cells = [
        "".join(c["source"])
        for c in json.loads(NOTEBOOK.read_text())["cells"]
        if c["cell_type"] == "code"
    ]
    render_cell, compile_cell = locate(cells)
    last = compile_cell if args.dry_run else len(cells) - 1
    namespace: dict = {"__name__": "__main__"}

    for index, source in enumerate(cells[: last + 1]):
        if index == render_cell:
            continue
        print(f"\n\033[90m--- cell {index} ---\033[0m", flush=True)
        try:
            exec(compile(source, f"<cell {index}>", "exec", dont_inherit=True), namespace)
        except Exception:
            print(f"\n\033[31mCELL {index} FAILED\033[0m")
            traceback.print_exc()
            return 1

    if args.dry_run:
        print("\n\033[32mgraph compiles; inputs load; no API calls made\033[0m")
    else:
        print("\n\033[32mfull run complete - see outputs/\033[0m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
