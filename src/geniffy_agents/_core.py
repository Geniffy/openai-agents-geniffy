"""What the memory and the tools share: whose memory a run is for, a client to reach it, and the words of an
input item whichever shape the Agents SDK hands it in."""
from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable, Dict, Optional, Tuple, Union

Space = Union[str, None, Callable[[Any], Optional[str]]]

_NO_SPACE = (
    "Geniffy needs a space: the user this is for, such as space=lambda user: f\"user_{user.id}\", read from "
    "the context you pass to Runner.run. Pass space=None only for your own memory, never for your users' data."
)


def check_space(space: Any) -> Optional[str]:
    """A space as given: None for your own memory, or a name. A blank name is refused rather than read as
    your own memory, because it is what an empty user id looks like, and that user would land in it."""
    if space is None:
        return None
    if not isinstance(space, str):
        raise TypeError(f"space must be a string (or None for your own memory), not {type(space).__name__}")
    if not space.strip():
        raise ValueError("space is blank: it would be your own memory, not a user's. " + _NO_SPACE)
    return space


def require_space(space: Any) -> None:
    if space is ...:
        raise TypeError(_NO_SPACE)
    if not callable(space):
        check_space(space)


def space_for(space: Space, context: Any) -> Optional[str]:
    """The space a run is for: the string, or what the function says for this run's context. A function names
    a user every time: None from it is refused, as a blank name is. For your own memory, pass space=None."""
    if callable(space):
        named = space(context)
        if named is None:
            raise ValueError("space returned None for this run, so there is no user to read or save for. "
                             "For your own memory, pass space=None itself rather than a function.")
        if not isinstance(named, str):
            raise TypeError(f"space must return a string, not {type(named).__name__}")
        return check_space(named)
    return space


# One AsyncGeniffy per event loop when no client is passed: a client's connections belong to the loop that
# opened them, so a server on one loop keeps one, and a script that calls asyncio.run per job gets one per run.
_shared: Dict[int, Tuple[asyncio.AbstractEventLoop, Any]] = {}


def default_client() -> Any:
    loop = asyncio.get_running_loop()
    held = _shared.get(id(loop))
    if held is None or held[0] is not loop:
        for key, (old, _) in list(_shared.items()):
            if old.is_closed():
                _shared.pop(key, None)
        from geniffy import AsyncGeniffy
        held = _shared[id(loop)] = (loop, AsyncGeniffy())
    return held[1]


def bound(client: Any, space: Optional[str]) -> Any:
    """The client (yours, or the shared one), pointed at one user's space, or your own memory for None."""
    client = client if client is not None else default_client()
    return client if space is None else client.space(space)


async def done(value: Any) -> Any:
    """An AsyncGeniffy's answer, or a Geniffy's: a sync client works too, but holds up the event loop."""
    return await value if inspect.isawaitable(value) else value


def _as_dict(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        return item
    dump = getattr(item, "model_dump", None)
    return dump() if callable(dump) else {}


def text_of(content: Any) -> str:
    """The words of an input item's content: a string, or parts with text in them."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [_as_dict(p) for p in content]
        return "\n".join(str(p.get("text") or "") for p in parts
                         if p.get("type") in ("input_text", "output_text", "text") and p.get("text")).strip()
    return ""


def last_user_text(items: Any) -> str:
    """The words of the last thing the user said: a string input, or the last user message among input items."""
    if isinstance(items, str):
        return items.strip()
    for item in reversed(list(items or [])):
        d = _as_dict(item)
        if d.get("role") == "user":
            words = text_of(d.get("content"))
            if words:
                return words
    return ""
