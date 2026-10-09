<p align="center">
  <a href="https://geniffy.com"><img src="https://geniffy.com/brand/geniffy-lockup-ink.png" alt="Geniffy" height="44"></a>
</p>

# Geniffy for the OpenAI Agents SDK

Give an agent built with the OpenAI Agents SDK a memory of each of your users. Before each model call, the
model gets the user's briefing: where things stand, what is due, the rules that apply, what happened, and what is
known that bears on their latest message. After the run, save it, tool calls included, into one memory for the
whole conversation. When nothing is known, the model is told so, and says so instead of guessing.

[![CI](https://github.com/Geniffy/openai-agents-geniffy/actions/workflows/ci.yml/badge.svg)](https://github.com/Geniffy/openai-agents-geniffy/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/openai-agents-geniffy)](https://pypi.org/project/openai-agents-geniffy/)
[![Docs](https://img.shields.io/badge/docs-docs.geniffy.com-1A1814)](https://docs.geniffy.com/integrations/openai-agents-sdk)

```bash
pip install openai-agents-geniffy
```

Set `GENIFFY_API_KEY` from **API keys** in the [Geniffy app](https://geniffy.com/app). Keep it on your server.

## The memory

```python
from dataclasses import dataclass

from agents import Agent, Runner, SQLiteSession
from geniffy_agents import GeniffyMemory


@dataclass
class User:
    id: str


memory = GeniffyMemory(space=lambda user: f"user_{user.id}")      # read from the context you run with
agent = Agent[User](name="Assistant", instructions="You are a helpful assistant.", model="gpt-5.5")


async def chat(user: User, chat_id: str, message: str) -> str:
    session = SQLiteSession(chat_id)                                # or your own store of the conversation
    result = await Runner.run(agent, message, context=user, session=session, run_config=memory.run_config())
    await memory.save(user, result, session=chat_id)               # the run, saved into that conversation
    return result.final_output
```

- **Before each model call**, the user's briefing goes in the instructions, after the agent's own: where things
  stand, what is due or was promised, the rules that apply, what happened, then what is known that bears on their
  latest message, each line dated. In a run with tools, Geniffy is asked once, not once per step.
- **`save()`** keeps the run that just finished: what the user said, each tool the agent called and what it
  returned, and the answer. A conversation is one memory however long it gets; the history a session hands the run
  is not sent again, and a save that fails goes with the conversation's next one. Without `session=`, each run is
  a memory of its own.
- **If Geniffy can't be reached**, the run goes on without memory, and `on_error` hears about it.

`run_config()` takes your own `RunConfig` and adds the memory to it; a `call_model_input_filter` you already
have runs after it.

Options: `project` (a string, or a function of the context) keeps the briefing to one project and labels what is
saved with it; `session` names the conversation the same way, so `save()` needs no `session=`; `briefing=False`
reads only what bears on the user's latest message; `budget_chars` is the most the briefing adds (6,000 characters
unless you say); `instructions` changes what the model is told about the memory; and `client` shares your own
`geniffy.AsyncGeniffy`. Without one, a client is made for each event loop and shared. An account without the
briefing switched on is given what bears on the latest message instead, and the briefing is tried again ten
minutes later. `remember(user, message, result)` still saves a single exchange, without the tool calls.

## Tools

To let the agent decide when to look something up or save it:

```python
from geniffy_agents import geniffy_tools

agent = Agent[User](name="Assistant", model="gpt-5.5",
                    tools=geniffy_tools(space=lambda user: f"user_{user.id}"))    # recall and remember
```

The tools read the user from the run's context, so the model never sees or chooses whose memory it reads.

## One space per user

`space` is required: the user this is for, as a string or a function of the context you run with. Each space
is a memory of its own, and nothing else can read it. `space=None` is your own memory, never your users' data,
and a blank space, or None from your function, is refused, so a user with no id never lands in your own memory.
When a user deletes their account, forget them with `AsyncGeniffy().forget_space(...)` from the
[geniffy](https://pypi.org/project/geniffy/) SDK.

## Develop

```bash
pip install -e ".[test]" && pytest     # through the SDK's own Runner, with its ScriptedModel, and the real
                                       # geniffy client against a stand-in for the API
```

## Security

Report a vulnerability to ops@geniffy.com, not in a public issue. See the
[security policy](https://github.com/Geniffy/.github/blob/main/SECURITY.md).
