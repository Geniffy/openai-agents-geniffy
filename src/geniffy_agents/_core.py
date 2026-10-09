"""What the memory and the tools share: whose memory a run is for, a client to reach it, and the words of an
input item whichever shape the Agents SDK hands it in."""
from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

Space = Union[str, None, Callable[[Any], Optional[str]]]
Named = Union[str, None, Callable[[Any], Optional[str]]]

# What Geniffy keeps of a message, so nothing larger is sent: a turn's words, a tool call's input and a tool's output.
TURN_CHARS, CALL_CHARS, RESULT_CHARS = 50_000, 2_000, 4_000

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
        held = _shared[id(loop)] = (loop, _made(AsyncGeniffy))
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


def named(value: Named, context: Any, what: str) -> Optional[str]:
    """A project or a session as given: a string, None, or what a function of the run's context says."""
    if callable(value):
        value = value(context)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{what} must be a string, not {type(value).__name__}")
    return value.strip() or None


def _said_by(d: Dict[str, Any]) -> Optional[str]:
    """The role of a message item ("user", "assistant", ...), or None for a tool call, its output or reasoning."""
    return d.get("role") if d.get("type", "message") == "message" else None


def turn_start(items: List[Any]) -> int:
    """Where the run that just finished starts among its input items: just after the last answer before the user's
    last message, so two messages sent at once are both in it. The history a session hands the run is before it."""
    roles = [_said_by(_as_dict(i)) for i in items]
    users = [i for i, r in enumerate(roles) if r == "user"]
    if not users:
        return 0
    for i in range(users[-1] - 1, -1, -1):
        if roles[i] == "assistant":
            return i + 1
    return 0


def _call(name: Any, args: Any, call_id: Any) -> Dict[str, Any]:
    if args is None:
        args = {}
    try:
        shown = args if isinstance(args, str) else json.dumps(args, default=str)
    except (TypeError, ValueError):
        shown = str(args)
    out: Dict[str, Any] = {"name": str(name or "tool")[:200],
                           "args": args if len(shown) <= CALL_CHARS else shown[:CALL_CHARS]}
    if call_id:
        out["id"] = str(call_id)[:200]
    return out


def _output(out: Any) -> str:
    return out.strip() if isinstance(out, str) else (text_of(out) if isinstance(out, list) else
                                                       json.dumps(out, default=str) if out is not None else "")


def as_sent(item: Any) -> List[Dict[str, Any]]:
    """An input item as Geniffy reads a conversation: what the user said, what the agent answered, each tool it
    called and what the tool returned. Instructions (system, developer) and the model's reasoning are left out; a
    hosted tool (web search, file search, code, a computer) is kept as the call, without what it brought back."""
    d = _as_dict(item)
    kind = d.get("type") or "message"
    if kind == "message":
        role = d.get("role")
        if role in ("user", "assistant"):
            return [{"role": role, "content": text_of(d.get("content"))[:TURN_CHARS]}]
        return []
    if kind in ("function_call", "custom_tool_call", "mcp_call"):
        call = _call(d.get("name"), d.get("arguments", d.get("input")), d.get("call_id") or d.get("id"))
        out = [{"role": "assistant", "content": "", "tool_calls": [call]}]
        if kind == "mcp_call" and d.get("output") is not None:     # an MCP call comes back in the same item
            out.append({"role": "tool", "content": _output(d.get("output"))[:RESULT_CHARS], "name": call["name"]})
        return out
    if kind in ("function_call_output", "custom_tool_call_output"):
        sent: Dict[str, Any] = {"role": "tool", "content": _output(d.get("output"))[:RESULT_CHARS]}
        if d.get("call_id"):
            sent["tool_call_id"] = str(d["call_id"])[:200]
        return [sent]
    if kind.endswith("_call"):
        args = next((d[k] for k in ("action", "queries", "code", "query") if d.get(k) is not None), None)
        return [{"role": "assistant", "content": "", "tool_calls": [_call(kind[:-5], args, d.get("id"))]}]
    return []


def sent_items(items: List[Any]) -> List[Dict[str, Any]]:
    out = [s for item in items for s in as_sent(item)]
    return [s for s in out if s.get("content") or s.get("tool_calls") or s["role"] == "tool"]


def title_of(sent: List[Dict[str, Any]]) -> Optional[str]:
    """What to call a saved conversation in the Geniffy app: the start of its first question, on one line."""
    for s in sent:
        words = " ".join(str(s.get("content") or "").split()) if s["role"] == "user" else ""
        if words:
            return words if len(words) <= 80 else words[:80].rsplit(" ", 1)[0] + "..."
    return None


def _made(cls: Any) -> Any:
    """A client of the geniffy SDK, named after this package, so the Requests page in the Geniffy app shows
    which integration made each call. A geniffy before 0.2.0 has no name to give, and makes a plain client."""
    from . import __version__
    try:
        return cls(integration=f"openai-agents-geniffy/{__version__}")
    except TypeError:
        return cls()
