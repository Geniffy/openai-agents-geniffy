"""GeniffyMemory with the real geniffy client, its calls answered by a stand-in for the API: what goes over the
wire for the briefing, for a run saved into its conversation, and for an account without the briefing switched on."""
from __future__ import annotations

import asyncio
import json
from typing import Any, List

import httpx
from agents import Agent, Runner
from agents.testing import ScriptedModel, assistant_message, function_call
from geniffy import AsyncGeniffy

from geniffy_agents import GeniffyMemory, __version__
from test_geniffy_agents import BRIEFING, CONTEXT, User, crm, per_user

SOURCE = {"id": "a" * 32, "kind": "note", "title": "Who signs the Lumen renewal?", "status": "reading"}


class Api:
    """The API, as far as the memory reaches it: the briefing (or not switched on), context and adding."""

    def __init__(self, briefing_on: bool = True):
        self.briefing_on = briefing_on
        self.calls: List[Any] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        self.calls.append((request.url.path, request.headers.get("x-geniffy-space"), body))
        assert f"openai-agents-geniffy/{__version__}" in request.headers["user-agent"]
        if request.url.path == "/v1/briefing":
            if not self.briefing_on:
                return httpx.Response(404, json={"error": {"code": "not_switched_on", "message": (
                    "The briefing, where things stand, episodes, lessons and intentions aren't switched on for "
                    "this memory yet.")}})
            return httpx.Response(200, json={"project": body.get("project"), "briefing": BRIEFING, "now": [],
                                             "due": [], "lessons": [], "episodes": [], "memories": []})
        if request.url.path == "/v1/context":
            return httpx.Response(200, json={"context": CONTEXT, "memories": [], "found": True})
        if request.url.path == "/v1/memories":
            return httpx.Response(201, json={"source": SOURCE})
        return httpx.Response(404, json={"error": {"code": "not_found", "message": "No such door."}})


def chat(api: Api) -> ScriptedModel:
    model = ScriptedModel([[function_call("crm", {"q": "Lumen"}, call_id="c1")],
                           [assistant_message("Priya Nair signs it.")]])
    agent = Agent[User](name="Assistant", model=model, tools=[crm])

    async def go() -> None:
        client = AsyncGeniffy("gnf_test", base_url="https://api.test",
                              integration=f"openai-agents-geniffy/{__version__}",
                              http_client=httpx.AsyncClient(transport=httpx.MockTransport(api)), max_retries=0)
        memory = GeniffyMemory(space=per_user, project="lumen", client=client)
        result = await Runner.run(agent, "Who signs the Lumen renewal?", context=User("1042"),
                                  run_config=memory.run_config())
        await memory.save(User("1042"), result, session="chat-7")

    asyncio.run(go())
    return model


def test_the_briefing_and_the_run_as_they_go_over_the_wire():
    api = Api()
    chat(api)
    assert api.calls == [
        ("/v1/briefing", "user_1042",
         {"project": "lumen", "cue": "Who signs the Lumen renewal?", "budget_chars": 6000}),
        ("/v1/memories", "user_1042", {"messages": [
            {"role": "user", "content": "Who signs the Lumen renewal?"},
            {"role": "assistant", "content": "", "tool_calls": [{"name": "crm", "args": '{"q":"Lumen"}', "id": "c1"}]},
            {"role": "tool", "content": "Lumen: renewal due 1 Nov.", "tool_call_id": "c1"},
            {"role": "assistant", "content": "Priya Nair signs it."}],
            "session": "chat-7", "labels": {"project": "lumen"}, "title": "Who signs the Lumen renewal?"}),
    ]


def test_an_account_without_the_briefing_reads_context_over_the_wire():
    api = Api(briefing_on=False)
    model = chat(api)
    assert [path for path, _, _ in api.calls] == ["/v1/briefing", "/v1/context", "/v1/memories"]
    assert CONTEXT in model.calls[0].system_instructions
