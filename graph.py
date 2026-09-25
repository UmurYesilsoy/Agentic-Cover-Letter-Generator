"""Thin re-export so LangGraph Studio / `langgraph dev` (pointed here via langgraph.json) has a
module to import. The pipeline itself - state, prompts, nodes, the compiled graph - lives entirely
in agent.py; this file exists only because pyproject.toml's py-modules build step and
langgraph.json both expect a standalone module, not a package attribute, to point at."""
from agent import graph
