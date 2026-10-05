"""Geniffy for the OpenAI Agents SDK: a memory of each of your users, for any agent.

    GeniffyMemory(space=...)           memory.run_config() puts what is known about the user in the instructions
                                       before each model call; memory.remember() saves the exchange
    geniffy_tools(space=...)           recall and remember, as function tools the agent calls itself

`space` names the user, as a string or a function of the context you run with, and is required, so one user's
memory is never written to another's; `space=None` is your own memory, never your users' data.
"""
from .memory import DEFAULT_INSTRUCTIONS, GeniffyMemory
from .tools import geniffy_tools

__all__ = ["GeniffyMemory", "geniffy_tools", "DEFAULT_INSTRUCTIONS"]
__version__ = "0.1.0"
