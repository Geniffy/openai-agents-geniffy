"""GeniffyMemory: a memory of each user for any agent built with the OpenAI Agents SDK."""
from __future__ import annotations

import json
import logging
import time
import uuid
from collections import OrderedDict
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Tuple

from agents import RunConfig
from agents.run_config import CallModelData, ModelInputData

from ._core import (TURN_CHARS, Named, Space, bound, done, last_user_text, named, require_space, sent_items,
                    space_for, title_of, turn_start)

logger = logging.getLogger("geniffy_agents")

DEFAULT_INSTRUCTIONS = (
    "What follows is this user's memory: what this app knows from before, each line with its date. "
    "Use it when it helps and don't recite it. If it doesn't cover something, say so instead of guessing."
)
# A briefing with nothing in it yet still says so: a model reads an empty memory as permission to invent.
NOTHING_YET = "Nothing is remembered about this user yet."
PER_CALL = 500          # the most messages Geniffy takes in one call
OFF_SECONDS = 600.0     # how long only what bears on the question is read after the briefing was found switched off
HELD = 256              # conversations whose failed saves are held for the next save

_Key = Tuple[Optional[str], Optional[str], str]


class _Recent:
    """What was just looked up, for a few seconds: a run calls the model once per step with the same question,
    and Geniffy is asked once. Keyed by space, project and question, so no user ever sees another's."""

    def __init__(self, seconds: float = 30.0, most: int = 512):
        self.seconds, self.most = seconds, most
        self._held: "OrderedDict[_Key, Tuple[float, Optional[str]]]" = OrderedDict()

    def get(self, key: _Key) -> Tuple[bool, Optional[str]]:
        hit = self._held.get(key)
        if hit and time.monotonic() - hit[0] < self.seconds:
            return True, hit[1]
        self._held.pop(key, None)
        return False, None

    def put(self, key: _Key, context: Optional[str]) -> None:
        self._held[key] = (time.monotonic(), context)
        while len(self._held) > self.most:
            self._held.popitem(last=False)

    def forget(self, space: Optional[str]) -> None:
        for key in [k for k in self._held if k[0] == space]:
            self._held.pop(key, None)


def _reply_text(result: Any) -> str:
    """The answer to save: a RunResult's final output, or the text itself. A structured output is saved as
    its JSON."""
    out = getattr(result, "final_output", result)
    if out is None:
        return ""
    if isinstance(out, str):
        return out.strip()
    dump = getattr(out, "model_dump_json", None)
    if callable(dump):
        return str(dump()).strip()
    try:
        return json.dumps(out, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(out).strip()


class GeniffyMemory:
    """Before each model call in a run, the user's briefing goes in the instructions, after the agent's own: where
    things stand, what is due, the rules that apply, what happened, then what is known that bears on their latest
    message. Once the run is done, save() keeps it, tool calls included, in one memory for the whole conversation.

        memory = GeniffyMemory(space=lambda user: f"user_{user.id}")    # from the context you run with

        result = await Runner.run(agent, message, context=user, session=session, run_config=memory.run_config())
        await memory.save(user, result, session=session.session_id)

    space        the user: a string, or a function of the run's context that names one every time (required;
                 space=None is your own memory, and a blank space is refused)
    project      the project the agent works on: a string, or a function of the run's context. The briefing keeps
                 to it, and what is saved is labelled with it. Left out: every project
    session      the conversation: a string, or a function of the run's context; save() can also be told. Left
                 out, each run is a memory of its own
    briefing     open with the briefing (default True); False reads only what is known that bears on the user's
                 latest message. An account without the briefing reads that, either way
    budget_chars the most the briefing adds to the instructions, in characters (500 to 40,000; default 6,000)
    instructions what the model is told about the memory it is given
    client       a geniffy.AsyncGeniffy to share (default: one per event loop, reading GENIFFY_API_KEY)
    on_error     called when Geniffy can't be reached; the run goes on without memory
    """

    def __init__(self, *, space: Space = ..., project: Named = None, session: Named = None, briefing: bool = True,
                 budget_chars: int = 6000, instructions: str = DEFAULT_INSTRUCTIONS, client: Any = None,
                 on_error: Optional[Callable[[Exception], None]] = None):
        require_space(space)
        self.space, self.project, self.session = space, project, session
        self.briefing, self.budget_chars = briefing, budget_chars
        self.instructions = instructions
        self.client = client
        self._on_error = on_error or (lambda e: logger.warning("Geniffy: %s", e))
        self._recent = _Recent()
        self._off_until = 0.0
        self._held: "OrderedDict[Tuple[Optional[str], str], List[Dict[str, Any]]]" = OrderedDict()

    # ── before each model call: the briefing ────────────────────────────────────────────────────────────────
    async def _read(self, mem: Any, project: Optional[str], question: str) -> Optional[str]:
        if self.briefing and time.monotonic() >= self._off_until:
            try:
                text = await done(mem.briefing(project=project, cue=question, budget_chars=self.budget_chars))
                return str(text or "") or NOTHING_YET
            except Exception as e:  # noqa: BLE001
                # not switched on for this key (404), or an API from before it (405): what bears on the question
                # is read instead, and the briefing is tried again in ten minutes
                if getattr(e, "status", None) not in (404, 405):
                    raise
                self._off_until = time.monotonic() + OFF_SECONDS
        return str(await done(mem.context(question))) if question else None

    async def _context(self, space: Optional[str], project: Optional[str], question: str) -> Optional[str]:
        seen, context = self._recent.get((space, project, question))
        if not seen:
            try:
                context = await self._read(bound(self.client, space), project, question)
            except Exception as e:  # noqa: BLE001 - the run goes on without memory, and isn't held up again
                self._on_error(e)
                context = None
            self._recent.put((space, project, question), context)
        return context

    async def filter(self, data: CallModelData[Any]) -> ModelInputData:
        """A RunConfig.call_model_input_filter: the model input, with the user's briefing in its instructions."""
        model, run = data.model_data, data.context
        context = await self._context(space_for(self.space, run), named(self.project, run, "project"),
                                      last_user_text(model.input))
        if not context:
            return model
        block = f"{self.instructions}\n\n<memory>\n{context}\n</memory>" if self.instructions else \
            f"<memory>\n{context}\n</memory>"
        return ModelInputData(input=model.input, instructions=f"{model.instructions}\n\n{block}"
                              if model.instructions else block)

    def run_config(self, base: Optional[RunConfig] = None) -> RunConfig:
        """A RunConfig for Runner.run that gives the model this memory. With `base`, that config with the memory
        added; a call_model_input_filter it already has runs after this one."""
        if base is None:
            return RunConfig(call_model_input_filter=self.filter)
        theirs = base.call_model_input_filter
        if theirs is None:
            return replace(base, call_model_input_filter=self.filter)

        async def both(data: CallModelData[Any]) -> ModelInputData:
            mine = await self.filter(data)
            return await done(theirs(replace(data, model_data=mine)))
        return replace(base, call_model_input_filter=both)

    # ── after the run: the conversation, saved as it goes ──────────────────────────────────────────────────
    async def _send(self, context: Any, sent: List[Dict[str, Any]], session: Optional[str]) -> None:
        """Into one memory for the conversation, 500 messages a call. A conversation's save that fails is held, and
        goes with its next save; without a conversation, each save is a memory of its own."""
        space = space_for(self.space, context)   # a missing user is the caller's to fix, never swallowed
        project = named(self.project, context, "project")
        session = session or named(self.session, context, "session")
        held = (space, session) if session else None
        batch = (self._held.pop(held, []) if held else []) + sent
        if not batch:
            return
        self._recent.forget(space)                # what was just said can bear on the next question
        name, done_up_to = session or f"run-{uuid.uuid4().hex}", 0
        try:
            for i in range(0, len(batch), PER_CALL):
                part = batch[i:i + PER_CALL]
                await done(bound(self.client, space).memories.add(
                    messages=part, session=name, title=title_of(batch),
                    labels={"project": project} if project else None))
                done_up_to = i + len(part)
        except Exception as e:  # noqa: BLE001
            if held:
                self._held[held] = batch[done_up_to:]
                while len(self._held) > HELD:
                    self._held.popitem(last=False)
            self._on_error(e)

    async def save(self, context: Any, result: Any, *, session: Optional[str] = None) -> None:
        """Save the run that just finished to the user's space: what they said, each tool the agent called and what it
        returned, and the answer, into one memory for the conversation (session=, or the session you made this
        memory with). The history a session hands the run is not sent again; a save that fails goes with the
        conversation's next one.

            result = await Runner.run(agent, message, context=user, session=session, run_config=memory.run_config())
            await memory.save(user, result, session=session.session_id)"""
        items = list(result.to_input_list())
        await self._send(context, sent_items(items[turn_start(items):]), session)

    async def remember(self, context: Any, message: Any, result: Any, *, session: Optional[str] = None) -> None:
        """Save one exchange to the user's space: what they said (a string, or the input items you ran with) and
        the answer (the RunResult, or its text), without the tools the agent called; save() keeps those too. The
        user's words become facts about the user."""
        asked, answer = last_user_text(message), _reply_text(result)
        if asked and answer:
            await self._send(context, [{"role": "user", "content": asked[:TURN_CHARS]},
                                       {"role": "assistant", "content": answer[:TURN_CHARS]}], session)
