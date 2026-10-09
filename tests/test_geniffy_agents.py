"""openai-agents-geniffy through the Agents SDK's own Runner, with its ScriptedModel (which keeps every call it
was given) and stand-ins for the Geniffy client: what the model sees, how often Geniffy is asked, what is saved
and where (tool calls included, one memory per conversation, the history a session hands a run never sent twice),
an account without the briefing, the tools, and a memory that is away."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, List

import pytest
from agents import Agent, RunConfig, Runner, SQLiteSession, function_tool
from agents.run_config import ModelInputData
from agents.testing import ScriptedModel, assistant_message, function_call

from geniffy_agents import GeniffyMemory, geniffy_tools
from geniffy_agents import _core

CONTEXT = "- Priya Nair signs the Lumen renewal.  [Call with Priya, 1 Oct 2026]"
BRIEFING = ("Where things stand (lumen), as of 2 Oct:\n- Next: send Priya the renewal\n"
            "What is known (facts):\n- [2026-10-01] Priya Nair signs the Lumen renewal.")


@dataclass
class User:
    id: str


class Status(Exception):
    """An error from the API, with its status, as the geniffy SDK raises them."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class Mem:
    """One user's memory, as client.space(...) gives it: records each call, with the space it was for."""

    def __init__(self, client: "Client", space: Any):
        self.client, self.log, self.space_id = client, client.log, space
        self.memories = self

    def _check(self) -> None:
        if self.client.fail:
            raise ConnectionError("Geniffy is away")

    async def briefing(self, *, project: Any = None, cue: str = "", budget_chars: int = 6000) -> str:
        self._check()
        self.log["briefing"].append((self.space_id, project, cue))
        if isinstance(self.client.briefing, Exception):
            raise self.client.briefing
        return self.client.briefing

    async def context(self, question: str) -> str:
        self._check()
        self.log["context"].append((self.space_id, question))
        return CONTEXT

    async def add(self, text: Any = None, *, messages: Any = None, session: Any = None, title: Any = None,
                  labels: Any = None) -> dict:
        self._check()
        if messages is None:
            self.log["add"].append((self.space_id, text))
            return {"id": "src_1"}
        if self.client.saves_fail:
            self.client.saves_fail -= 1
            raise Status(502, "Your memory didn't answer. Try again in a minute.")
        self.log["session"].append({"space": self.space_id, "session": session, "title": title, "labels": labels,
                                    "messages": messages})
        return {"id": "src_1"}


class Client:
    """A stand-in for geniffy.AsyncGeniffy. briefing is what the briefing says, or the error it raises; saves_fail is
    how many of the next saves fail."""

    def __init__(self, fail: bool = False, briefing: Any = BRIEFING, saves_fail: int = 0):
        self.log: dict = {"briefing": [], "context": [], "add": [], "session": []}
        self.fail, self.briefing, self.saves_fail = fail, briefing, saves_fail

    def space(self, space: str) -> Mem:
        return Mem(self, space)


NOTHING_ASKED = {"briefing": [], "context": [], "add": [], "session": []}


def per_user(user: User) -> str:
    return f"user_{user.id}"


def run(agent: Agent, message: Any, user: User, config: Any) -> Any:
    return asyncio.run(Runner.run(agent, message, context=user, run_config=config))


@function_tool
async def crm(q: str) -> str:
    """Look something up in the CRM.

    Args:
        q: what to look up
    """
    return "Lumen: renewal due 1 Nov."


def test_the_model_is_given_the_briefing_after_the_agents_own_instructions():
    client = Client()
    model = ScriptedModel([[assistant_message("Priya Nair signs it.")]])
    agent = Agent[User](name="Assistant", instructions="You are a helpful assistant.", model=model)
    memory = GeniffyMemory(space=per_user, client=client)

    result = run(agent, "Who signs the Lumen renewal?", User("1042"), memory.run_config())

    assert result.final_output == "Priya Nair signs it."
    system = model.calls[0].system_instructions
    assert system.startswith("You are a helpful assistant.\n\n") and f"<memory>\n{BRIEFING}\n</memory>" in system
    assert "each line with its date" in system
    assert client.log["briefing"] == [("user_1042", None, "Who signs the Lumen renewal?")]
    assert client.log["context"] == [], "the briefing holds what is known that bears on the question"
    assert client.log["session"] == [], "nothing is saved until save()"


def test_geniffy_is_asked_once_however_many_steps_a_run_takes():
    client = Client()
    model = ScriptedModel([[function_call("crm", {"q": "Lumen"}, call_id="c1")],
                           [function_call("crm", {"q": "owner"}, call_id="c2")],
                           [assistant_message("Priya Nair signs it.")]])
    agent = Agent[User](name="Assistant", instructions="You are a helpful assistant.", model=model, tools=[crm])
    run(agent, "Who signs the Lumen renewal?", User("1042"), GeniffyMemory(space=per_user, client=client).run_config())

    assert len(model.calls) == 3
    assert all(BRIEFING in c.system_instructions for c in model.calls), "every step sees the memory"
    assert len(client.log["briefing"]) == 1, "and Geniffy is asked once"


def test_save_keeps_the_whole_run_with_its_tool_calls_and_a_session_sends_no_history_twice():
    client = Client()
    model = ScriptedModel([[function_call("crm", {"q": "Lumen"}, call_id="c1")],
                           [assistant_message("Priya Nair signs it, due 1 Nov.")],
                           [assistant_message("Send it Monday.")]])
    agent = Agent[User](name="Assistant", instructions="Be brief.", model=model, tools=[crm])
    memory = GeniffyMemory(space=per_user, project="lumen", client=client)
    user = User("1042")

    async def chat() -> None:
        session = SQLiteSession("chat-7")
        for message in ("Who signs the Lumen renewal?", "When should I send it?"):
            result = await Runner.run(agent, message, context=user, session=session, run_config=memory.run_config())
            await memory.save(user, result, session=session.session_id)

    asyncio.run(chat())
    first, second = client.log["session"]
    assert (first["space"], first["session"], first["labels"]) == ("user_1042", "chat-7", {"project": "lumen"})
    assert first["title"] == "Who signs the Lumen renewal?"
    assert first["messages"] == [
        {"role": "user", "content": "Who signs the Lumen renewal?"},
        {"role": "assistant", "content": "", "tool_calls": [{"name": "crm", "args": '{"q":"Lumen"}', "id": "c1"}]},
        {"role": "tool", "content": "Lumen: renewal due 1 Nov.", "tool_call_id": "c1"},
        {"role": "assistant", "content": "Priya Nair signs it, due 1 Nov."},
    ], "what the agent did is how Geniffy learns what happened"
    assert second["session"] == "chat-7" and second["messages"] == [
        {"role": "user", "content": "When should I send it?"}, {"role": "assistant", "content": "Send it Monday."}], \
        "the history the session handed the run is not sent again"
    assert [b[1] for b in client.log["briefing"]] == ["lumen", "lumen"]


def test_a_session_from_the_run_context_and_a_save_that_fails_goes_with_the_next():
    heard: List[Exception] = []
    client = Client(saves_fail=1)
    model = ScriptedModel([[assistant_message("Priya Nair signs it.")], [assistant_message("In March.")]])
    agent = Agent[User](name="Assistant", model=model)
    memory = GeniffyMemory(space=per_user, session=lambda user: f"chat-{user.id}", client=client,
                           on_error=heard.append)
    user = User("1042")

    async def chat() -> None:
        history: List[Any] = []
        for message in ("Who signs the Lumen renewal?", "And when is it due?"):
            result = await Runner.run(agent, history + [{"role": "user", "content": message}], context=user,
                                      run_config=memory.run_config())
            await memory.save(user, result)
            history = result.to_input_list()

    asyncio.run(chat())
    assert [e.status for e in heard] == [502]
    [saved] = client.log["session"]
    assert saved["session"] == "chat-1042"
    assert [m["content"] for m in saved["messages"]] == [
        "Who signs the Lumen renewal?", "Priya Nair signs it.", "And when is it due?", "In March."]


def test_without_a_session_each_run_is_a_memory_of_its_own():
    client = Client()
    model = ScriptedModel([[assistant_message("Noted.")]])
    agent = Agent[User](name="Assistant", model=model)
    memory = GeniffyMemory(space=per_user, client=client)

    async def chat() -> None:
        result = await Runner.run(agent, "I moved to Pune.", context=User("1042"), run_config=memory.run_config())
        await memory.save(User("1042"), result)

    asyncio.run(chat())
    assert client.log["session"][0]["session"].startswith("run-")


def test_remember_saves_one_exchange_to_that_users_space():
    client = Client()
    model = ScriptedModel([[assistant_message("Noted: WhatsApp, not email.")]])
    agent = Agent[User](name="Assistant", instructions="Be brief.", model=model)
    memory = GeniffyMemory(space=per_user, client=client)

    async def chat() -> None:
        result = await Runner.run(agent, "Reach me on WhatsApp, not email.", context=User("1042"),
                                  run_config=memory.run_config())
        await memory.remember(User("1042"), "Reach me on WhatsApp, not email.", result)

    asyncio.run(chat())
    assert [(s["space"], s["messages"]) for s in client.log["session"]] == [
        ("user_1042", [{"role": "user", "content": "Reach me on WhatsApp, not email."},
                       {"role": "assistant", "content": "Noted: WhatsApp, not email."}])]

    items = [{"role": "user", "content": [{"type": "input_text", "text": "And call after 6."}]}]
    asyncio.run(memory.remember(User("7"), items, "Will do.", session="chat-7"))
    assert (client.log["session"][-1]["space"], client.log["session"][-1]["session"]) == ("user_7", "chat-7")
    assert client.log["session"][-1]["messages"] == [{"role": "user", "content": "And call after 6."},
                                                     {"role": "assistant", "content": "Will do."}]
    asyncio.run(memory.remember(User("7"), "", "nothing asked"))
    asyncio.run(memory.remember(User("7"), "asked", ""))
    assert len(client.log["session"]) == 2, "an exchange with a side missing is not saved"


def test_a_memory_that_is_away_never_stops_the_run():
    heard: List[Exception] = []
    model = ScriptedModel([[function_call("crm", {"q": "x"}, call_id="c1")], [assistant_message("I don't know.")]])
    agent = Agent[User](name="Assistant", instructions="You are a helpful assistant.", model=model, tools=[crm])
    memory = GeniffyMemory(space=per_user, client=Client(fail=True), on_error=heard.append)

    async def chat() -> Any:
        result = await Runner.run(agent, "Who signs the Lumen renewal?", context=User("1042"),
                                  run_config=memory.run_config())
        await memory.save(User("1042"), result)
        return result

    assert asyncio.run(chat()).final_output == "I don't know."
    assert all(c.system_instructions == "You are a helpful assistant." for c in model.calls)
    assert [type(e) for e in heard] == [ConnectionError, ConnectionError], \
        "the lookup heard once, not at every step of the run, and the save"


def test_without_the_briefing_switched_on_it_reads_what_is_known_and_tries_again_later():
    client = Client(briefing=Status(404, "The briefing isn't switched on for this memory yet."))
    memory = GeniffyMemory(space=per_user, client=client, on_error=lambda e: pytest.fail(f"not an error: {e}"))
    model = ScriptedModel([[assistant_message("Priya Nair signs it.")], [assistant_message("Pune.")],
                           [assistant_message("In March.")]])
    agent = Agent[User](name="Assistant", model=model)
    run(agent, "Who signs the Lumen renewal?", User("1042"), memory.run_config())
    run(agent, "Where do I live?", User("1042"), memory.run_config())
    assert f"<memory>\n{CONTEXT}\n</memory>" in model.calls[0].system_instructions
    assert len(client.log["briefing"]) == 1, "asked once, then not for ten minutes"
    assert [q for _, q in client.log["context"]] == ["Who signs the Lumen renewal?", "Where do I live?"]

    client.briefing, memory._off_until = BRIEFING, 0.0       # ten minutes on, and it is switched on
    run(agent, "And when is it due?", User("1042"), memory.run_config())
    assert BRIEFING in model.calls[2].system_instructions


def test_a_briefing_that_fails_another_way_is_an_error_not_a_reason_to_read_something_else():
    heard: List[Exception] = []
    client = Client(briefing=Status(502, "Your memory didn't answer."))
    model = ScriptedModel([[assistant_message("Hello.")]])
    agent = Agent[User](name="Assistant", instructions="Hi.", model=model)
    run(agent, "Hello", User("1042"), GeniffyMemory(space=per_user, client=client, on_error=heard.append).run_config())
    assert [e.status for e in heard] == [502] and client.log["context"] == []
    assert model.calls[0].system_instructions == "Hi."


def test_an_empty_briefing_still_says_nothing_is_remembered_yet_and_briefing_false_reads_the_question():
    model = ScriptedModel([[assistant_message("I don't know yet.")], [assistant_message("Priya.")]])
    agent = Agent[User](name="Assistant", model=model)
    run(agent, "What is my name?", User("1042"), GeniffyMemory(space=per_user, client=Client(briefing="")).run_config())
    assert "<memory>\nNothing is remembered about this user yet.\n</memory>" in model.calls[0].system_instructions

    client = Client()
    run(agent, "Who signs it?", User("1042"), GeniffyMemory(space=per_user, client=client, briefing=False).run_config())
    assert client.log["briefing"] == [] and client.log["context"] == [("user_1042", "Who signs it?")]


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
    assert client.log == NOTHING_ASKED


def test_space_none_is_your_own_memory():
    log: dict = {"briefing": []}

    class Own:
        def space(self, space: str) -> Any:
            raise AssertionError("space=None must not pick a space")

        async def briefing(self, **kwargs: Any) -> str:
            log["briefing"].append(kwargs["cue"])
            return BRIEFING

    model = ScriptedModel([[assistant_message("Done.")]])
    agent = Agent[User](name="Assistant", model=model)
    run(agent, "What did I decide?", User("1"), GeniffyMemory(space=None, client=Own()).run_config())
    assert log["briefing"] == ["What did I decide?"]
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
    assert BRIEFING in seen[0], "your filter sees the memory already in"
    assert model.calls[0].system_instructions.endswith("\n\nP.S.")


def test_items_are_sent_as_geniffy_reads_them_and_no_larger_than_it_keeps():
    assert _core.as_sent({"type": "reasoning", "summary": []}) == []
    assert _core.as_sent({"role": "system", "content": "Be brief."}) == []
    assert _core.as_sent({"type": "function_call_output", "call_id": "c9", "output": "x" * 10_000})[0]["content"] \
        == "x" * 4_000
    assert _core.as_sent({"type": "web_search_call", "id": "ws1", "status": "completed",
                          "action": {"type": "search", "query": "Lumen renewal"}}) == [
        {"role": "assistant", "content": "",
         "tool_calls": [{"name": "web_search", "args": {"type": "search", "query": "Lumen renewal"}, "id": "ws1"}]}]
    assert _core.as_sent({"type": "mcp_call", "id": "m1", "name": "lookup", "arguments": "{}",
                          "output": "found", "server_label": "crm"}) == [
        {"role": "assistant", "content": "", "tool_calls": [{"name": "lookup", "args": "{}", "id": "m1"}]},
        {"role": "tool", "content": "found", "name": "lookup"}]
    big = _core.as_sent({"type": "function_call", "call_id": "c10", "name": "write", "arguments": "y" * 9_000})
    assert len(big[0]["tool_calls"][0]["args"]) == 2_000


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
    assert len(made) == 1 and made[0].log["briefing"] == [("user_1", None, "hi"), ("user_2", None, "hi")]
    asyncio.run(twice("hello again"))
    assert len(made) == 2 and len(_core._shared) == 1, "a new loop gets its own; the closed one's is let go"
