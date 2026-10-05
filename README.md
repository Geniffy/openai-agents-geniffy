<p align="center">
  <a href="https://geniffy.com"><img src="https://geniffy.com/brand/geniffy-lockup-ink.png" alt="Geniffy" height="44"></a>
</p>

# Geniffy for the OpenAI Agents SDK

Give an agent built with the OpenAI Agents SDK a memory of each of your users. Before each model call, the
model is told what is known about the user that bears on their latest message, each line with where it came
from; after the run, save the exchange. When nothing is known, the model is told so, and says so instead of
guessing.

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

from agents import Agent, Runner
from geniffy_agents import GeniffyMemory


@dataclass
class User:
    id: str


memory = GeniffyMemory(space=lambda user: f"user_{user.id}")      # read from the context you run with
agent = Agent[User](name="Assistant", instructions="You are a helpful assistant.", model="gpt-5.5")


async def chat(user: User, message: str) -> str:
    result = await Runner.run(agent, message, context=user, run_config=memory.run_config())
    await memory.remember(user, message, result)                   # the exchange, saved to that user's space
    return result.final_output
```

- **Before each model call**, what is known that bears on the user's latest message goes in the instructions,
  after the agent's own. In a run with tools, Geniffy is asked once, not once per step.
- **`remember()`** saves the exchange: the user's words become facts about the user.
- **If Geniffy can't be reached**, the run goes on without memory, and `on_error` hears about it.

`run_config()` takes your own `RunConfig` and adds the memory to it; a `call_model_input_filter` you already
have runs after it. Options: `instructions` to change what the model is told about the memory, and `client` to
share your own `geniffy.AsyncGeniffy`. Without one, a client is made for each event loop and shared.

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
pip install -e ".[test]" && pytest     # through the SDK's own Runner, with its ScriptedModel
```

## Security

Report a vulnerability to ops@geniffy.com, not in a public issue. See the
[security policy](https://github.com/Geniffy/.github/blob/main/SECURITY.md).
