"""openai-agents-geniffy through the Agents SDK's own Runner, with its ScriptedModel (which keeps every call it
was given) and stand-ins for the Geniffy client: what the model sees, how often Geniffy is asked, what is saved
and where, the tools, and a memory that is away."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, List

import pytest
from agents import Agent, RunConfig, Runner, function_tool
from agents.run_config import ModelInputData
from agents.testing import ScriptedModel, assistant_message, function_call

from geniffy_agents import GeniffyMemory, geniffy_tools
from geniffy_agents import _core

CONTEXT = "- Priya Nair signs the Lumen renewal.  [Call with Priya, 1 Oct 2026]"


@dataclass
class User:
    id: str


class Mem:
    """One user's memory, as client.space(...) gives it: records each call, with the space it was for."""

    def __init__(self, log: dict, space: Any, fail: bool = False):
        self.log, self.space_id, self.fail = log, space, fail
        self.memories = self

    def _check(self) -> None:
        if self.fail:
            raise ConnectionError("Geniffy is away")

    async def context(self, question: str) -> str:
        self._check()
        self.log["context"].append((self.space_id, question))
        return CONTEXT

    async def add(self, text: Any = None, *, messages: Any = None) -> dict:
        self._check()
        self.log["add"].append((self.space_id, text if messages is None else messages))
        return {"id": "src_1"}


class Client:
    def __init__(self, fail: bool = False):
        self.log: dict = {"context": [], "add": []}
        self.fail = fail

    def space(self, space: str) -> Mem:
        return Mem(self.log, space, self.fail)


def per_user(user: User) -> str:
    return f"user_{user.id}"


def run(agent: Agent, message: Any, user: User, config: Any) -> Any:
    return asyncio.run(Runner.run(agent, message, context=user, run_config=config))


def test_the_model_is_told_what_is_known_after_the_agents_own_instructions():
    client = Client()
    model = ScriptedModel([[assistant_message("Priya Nair signs it.")]])
    agent = Agent[User](name="Assistant", instructions="You are a helpful assistant.", model=model)
    memory = GeniffyMemory(space=per_user, client=client)

    result = run(agent, "Who signs the Lumen renewal?", User("1042"), memory.run_config())

    assert result.final_output == "Priya Nair signs it."
    system = model.calls[0].system_instructions
    assert system.startswith("You are a helpful assistant.\n\n") and f"<memory>\n{CONTEXT}\n</memory>" in system
    assert "each line with where it came from" in system
    assert client.log["context"] == [("user_1042", "Who signs the Lumen renewal?")]
    assert client.log["add"] == [], "nothing is saved until remember()"


def test_geniffy_is_asked_once_however_many_steps_a_run_takes():
    client = Client()
    lookups: List[str] = []

    @function_tool
    async def lookup(q: str) -> str:
        """Look something up in the CRM.

        Args:
            q: what to look up
        """
        lookups.append(q)
        return "Lumen: renewal due 1 Nov."

    model = ScriptedModel([[function_call("lookup", {"q": "Lumen"}, call_id="c1")],
                           [function_call("lookup", {"q": "owner"}, call_id="c2")],
                           [assistant_message("Priya Nair signs it.")]])
    agent = Agent[User](name="Assistant", instructions="You are a helpful assistant.", model=model, tools=[lookup])
    run(agent, "Who signs the Lumen renewal?", User("1042"), GeniffyMemory(space=per_user, client=client).run_config())

    assert lookups == ["Lumen", "owner"] and len(model.calls) == 3
    assert all(CONTEXT in c.system_instructions for c in model.calls), "every step sees the memory"
    assert len(client.log["context"]) == 1, "and Geniffy is asked once"


def test_remember_saves_the_exchange_to_that_users_space():
    client = Client()
    model = ScriptedModel([[assistant_message("Noted: WhatsApp, not email.")]])
    agent = Agent[User](name="Assistant", instructions="Be brief.", model=model)
    memory = GeniffyMemory(space=per_user, client=client)

    async def chat() -> None:
        result = await Runner.run(agent, "Reach me on WhatsApp, not email.", context=User("1042"),
                                  run_config=memory.run_config())
        await memory.remember(User("1042"), "Reach me on WhatsApp, not email.", result)

    asyncio.run(chat())
    assert client.log["add"] == [("user_1042", [{"role": "user", "content": "Reach me on WhatsApp, not email."},
                                                {"role": "assistant", "content": "Noted: WhatsApp, not email."}])]

    items = [{"role": "user", "content": [{"type": "input_text", "text": "And call after 6."}]}]
    asyncio.run(memory.remember(User("7"), items, "Will do."))
    assert client.log["add"][-1] == ("user_7", [{"role": "user", "content": "And call after 6."},
                                                {"role": "assistant", "content": "Will do."}])
    asyncio.run(memory.remember(User("7"), "", "nothing asked"))
    asyncio.run(memory.remember(User("7"), "asked", ""))
    assert len(client.log["add"]) == 2, "an exchange with a side missing is not saved"


def test_a_memory_that_is_away_never_stops_the_run():
    heard: List[Exception] = []
    model = ScriptedModel([[function_call("lookup", {"q": "x"}, call_id="c1")], [assistant_message("I don't know.")]])

    @function_tool
    async def lookup(q: str) -> str:
        """Look something up.

        Args:
            q: what to look up
        """
        return "nothing"

    agent = Agent[User](name="Assistant", instructions="You are a helpful assistant.", model=model, tools=[lookup])
    memory = GeniffyMemory(space=per_user, client=Client(fail=True), on_error=heard.append)
    result = run(agent, "Who signs the Lumen renewal?", User("1042"), memory.run_config())

    assert result.final_output == "I don't know."
    assert all(c.system_instructions == "You are a helpful assistant." for c in model.calls)
    assert [type(e) for e in heard] == [ConnectionError], "heard once, not at every step of the run"


def test_a_space_is_required_and_a_missing_user_is_refused():
    with pytest.raises(TypeError, match="needs a space"):
        GeniffyMemory()
    with pytest.raises(TypeError, match="needs a space"):
        geniffy_tools()
    for blank in ("", "   "):
        with pytest.raises(ValueError, match="blank"):
            GeniffyMemory(space=blank)
        with pytest.raises(ValueError, match="blank"):
            geniffy_tools(space=blank)

    client = Client()
    model = ScriptedModel([[assistant_message("Hello.")]])
    agent = Agent[User](name="Assistant", instructions="Hi.", model=model)
    memory = GeniffyMemory(space=lambda user: None, client=client)
    with pytest.raises(ValueError, match="returned None"):
        run(agent, "Hello", User("1042"), memory.run_config())
    with pytest.raises(ValueError, match="returned None"):
        asyncio.run(memory.remember(User("1042"), "Hello", "Hi"))
    assert client.log == {"context": [], "add": []}


def test_space_none_is_your_own_memory():
    log: dict = {"context": []}

    class Own:
        def space(self, space: str) -> Any:
            raise AssertionError("space=None must not pick a space")

        async def context(self, question: str) -> str:
            log["context"].append(question)
            return CONTEXT

    model = ScriptedModel([[assistant_message("Done.")]])
    agent = Agent[User](name="Assistant", model=model)
    run(agent, "What did I decide?", User("1"), GeniffyMemory(space=None, client=Own()).run_config())
    assert log["context"] == ["What did I decide?"]
    assert model.calls[0].system_instructions.startswith("What follows is this user's memory"), \
        "an agent with no instructions of its own gets the memory alone"


def test_your_own_run_config_and_input_filter_are_kept():
    client = Client()
    seen: List[str] = []

    def mine(data: Any) -> ModelInputData:
        seen.append(data.model_data.instructions)
        return ModelInputData(input=data.model_data.input, instructions=data.model_data.instructions + "\n\nP.S.")

    model = ScriptedModel([[assistant_message("Done.")]])
    agent = Agent[User](name="Assistant", instructions="Base.", model=model)
    config = GeniffyMemory(space=per_user, client=client).run_config(
        RunConfig(workflow_name="support", call_model_input_filter=mine))
    assert config.workflow_name == "support"
    run(agent, "Who signs?", User("1042"), config)
    assert CONTEXT in seen[0], "your filter sees the memory already in"
    assert model.calls[0].system_instructions.endswith("\n\nP.S.")


def test_the_tools_read_the_user_from_the_run_context():
    client = Client()
    recall, remember = geniffy_tools(space=per_user, client=client)
    assert (recall.name, remember.name) == ("recall", "remember")
    assert "When nothing is known, it says so" in recall.description
    assert "Never passwords" in remember.description
    assert "space" not in str(recall.params_json_schema), "the model never sees or chooses whose memory it reads"

    model = ScriptedModel([[function_call("recall", {"query": "Lumen renewal"}, call_id="c1")],
                           [function_call("remember", {"text": "Prefers WhatsApp."}, call_id="c2")],
                           [assistant_message("Priya Nair signs it.")]])
    agent = Agent[User](name="Assistant", instructions="Use recall.", model=model, tools=[recall, remember])
    result = asyncio.run(Runner.run(agent, "Who signs the Lumen renewal?", context=User("1042")))

    assert result.final_output == "Priya Nair signs it."
    outputs = [i for i in model.calls[1].input if isinstance(i, dict) and i.get("type") == "function_call_output"]
    assert CONTEXT in str(outputs[0]["output"]), "the model gets what recall found"
    assert client.log["context"] == [("user_1042", "Lumen renewal")]
    assert client.log["add"] == [("user_1042", "Prefers WhatsApp.")]


def test_without_a_client_one_is_made_for_each_event_loop(monkeypatch):
    import geniffy
    made: List[Any] = []

    class Fake(Client):
        def __init__(self, *a: Any, **kw: Any):
            super().__init__()
            made.append(self)

    monkeypatch.setattr(geniffy, "AsyncGeniffy", Fake)
    monkeypatch.setattr(_core, "_shared", {})
    memory = GeniffyMemory(space=per_user)

    async def twice(question: str) -> None:
        for uid in ("1", "2"):
            model = ScriptedModel([[assistant_message("Hi.")]])
            await Runner.run(Agent[User](name="A", model=model), question, context=User(uid),
                             run_config=memory.run_config())

    asyncio.run(twice("hi"))
    assert len(made) == 1 and made[0].log["context"] == [("user_1", "hi"), ("user_2", "hi")]
    asyncio.run(twice("hello again"))
    assert len(made) == 2 and len(_core._shared) == 1, "a new loop gets its own; the closed one's is let go"
