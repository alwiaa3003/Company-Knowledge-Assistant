"""ADK app entry point for the single-agent baseline (one LlmAgent with both tools)."""

from kl_agents.builders import build_single_agent

root_agent = build_single_agent()
