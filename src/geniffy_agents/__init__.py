"""Geniffy for the OpenAI Agents SDK: a memory of each of your users, for any agent.

    GeniffyMemory(space=...)           memory.run_config() puts the user's briefing in the instructions before
                                       each model call; memory.save() keeps the run, tool calls included, in one
                                       memory for the conversation
    geniffy_tools(space=...)           recall and remember, as function tools the agent calls itself

`space` names the user, as a string or a function of the context you run with, and is required, so one user's
memory is never written to another's; `space=None` is your own memory, never your users' data.
"""
from .memory import DEFAULT_INSTRUCTIONS, GeniffyMemory
from .tools import geniffy_tools

__all__ = ["GeniffyMemory", "geniffy_tools", "DEFAULT_INSTRUCTIONS"]
__version__ = "0.2.0"
