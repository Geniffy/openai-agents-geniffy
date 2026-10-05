"""GeniffyMemory: a memory of each user for any agent built with the OpenAI Agents SDK."""
from __future__ import annotations

import json
import logging
import time
from collections import OrderedDict
from dataclasses import replace
from typing import Any, Callable, Optional, Tuple

from agents import RunConfig
from agents.run_config import CallModelData, ModelInputData

from ._core import Space, bound, done, last_user_text, require_space, space_for

logger = logging.getLogger("geniffy_agents")

DEFAULT_INSTRUCTIONS = (
    "What follows is this user's memory: what they told this app before, each line with where it came from. "
    "Use it when it helps and don't recite it. If it doesn't cover something, say so instead of guessing."
)


class _Recent:
    """What was just looked up, for a few seconds: a run calls the model once per step with the same question,
    and Geniffy is asked once. Keyed by space and question, so no user ever sees another's."""

    def __init__(self, seconds: float = 30.0, most: int = 512):
        self.seconds, self.most = seconds, most
        self._held: "OrderedDict[Tuple[Optional[str], str], Tuple[float, Optional[str]]]" = OrderedDict()

    def get(self, key: Tuple[Optional[str], str]) -> Tuple[bool, Optional[str]]:
        hit = self._held.get(key)
        if hit and time.monotonic() - hit[0] < self.seconds:
            return True, hit[1]
        self._held.pop(key, None)
        return False, None

    def put(self, key: Tuple[Optional[str], str], context: Optional[str]) -> None:
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
    """Before each model call in a run, what Geniffy knows about the user that bears on their latest message
    goes in the instructions, after the agent's own. Save each exchange with remember() once the run is done.

        memory = GeniffyMemory(space=lambda user: f"user_{user.id}")    # from the context you run with

        result = await Runner.run(agent, message, context=user, run_config=memory.run_config())
        await memory.remember(user, message, result)

    space        the user: a string, or a function of the run's context that names one every time (required;
                 space=None is your own memory, and a blank space is refused)
    instructions what the model is told about the memory it is given
    client       a geniffy.AsyncGeniffy to share (default: one per event loop, reading GENIFFY_API_KEY)
    on_error     called when Geniffy can't be reached; the run goes on without memory
    """

    def __init__(self, *, space: Space = ..., instructions: str = DEFAULT_INSTRUCTIONS, client: Any = None,
                 on_error: Optional[Callable[[Exception], None]] = None):
        require_space(space)
        self.space = space
        self.instructions = instructions
        self.client = client
        self._on_error = on_error or (lambda e: logger.warning("Geniffy: %s", e))
        self._recent = _Recent()

    async def _context(self, space: Optional[str], question: str) -> Optional[str]:
        seen, context = self._recent.get((space, question))
        if not seen:
            try:
                context = str(await done(bound(self.client, space).context(question)))
            except Exception as e:  # noqa: BLE001 - the run goes on without memory, and isn't held up again
                self._on_error(e)
                context = None
            self._recent.put((space, question), context)
        return context

    async def filter(self, data: CallModelData[Any]) -> ModelInputData:
        """A RunConfig.call_model_input_filter: the model input, with the user's memory in its instructions."""
        model = data.model_data
        question = last_user_text(model.input)
        if not question:
            return model
        context = await self._context(space_for(self.space, data.context), question)
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

    async def remember(self, context: Any, message: Any, result: Any) -> None:
        """Save one exchange to the user's space: what they said (a string, or the input items you ran with) and
        the answer (the RunResult, or its text). The user's words become facts about the user."""
        asked, answer = last_user_text(message), _reply_text(result)
        if not asked or not answer:
            return
        space = space_for(self.space, context)   # a missing user is the caller's to fix, never swallowed
        self._recent.forget(space)                # what was just said can bear on the next question
        try:
            await done(bound(self.client, space).memories.add(messages=[{"role": "user", "content": asked},
                                                                       {"role": "assistant", "content": answer}]))
        except Exception as e:  # noqa: BLE001
            self._on_error(e)
