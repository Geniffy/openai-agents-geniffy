"""recall and remember, as function tools for an agent that decides for itself when to look something up."""
from __future__ import annotations

from typing import Any, List

from agents import FunctionTool, RunContextWrapper, function_tool

from ._core import Space, bound, done, require_space, space_for


def geniffy_tools(*, space: Space = ..., client: Any = None) -> List[FunctionTool]:
    """[recall, remember], reading the user from the run's context, so the model never sees or chooses whose
    memory it reads.

        agent = Agent[User](name="Assistant", tools=geniffy_tools(space=lambda user: f"user_{user.id}"))
    """
    require_space(space)

    @function_tool(name_override="recall")
    async def recall(ctx: RunContextWrapper[Any], query: str) -> str:
        """Look up what is known about the user: their preferences, people, plans and what they said before,
        each line with where it came from. When nothing is known, it says so.

        Args:
            query: What to look up, in the user's own words.
        """
        return str(await done(bound(client, space_for(space, ctx.context)).context(query)))

    @function_tool(name_override="remember")
    async def remember(ctx: RunContextWrapper[Any], text: str) -> str:
        """Save something the user asked you to remember, in their words. Never passwords, keys or other secrets.

        Args:
            text: What to remember.
        """
        await done(bound(client, space_for(space, ctx.context)).memories.add(text))
        return "Saved."

    return [recall, remember]
