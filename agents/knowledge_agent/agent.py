"""ADK app entry point for `adk run` / `adk web` (https://adk.dev/get-started/python/).

ADK requires this folder layout: <agents_dir>/<app_name>/__init__.py (``from . import agent``) and
agent.py defining ``root_agent``. Building the agent here reads settings (.env at the project root,
via kb.config) - this is the one place where work happens at import time, because ADK needs it.
"""

from kl_agents.builders import build_hierarchy

root_agent = build_hierarchy()
