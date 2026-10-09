# Changelog

## 0.2.0

- **The briefing.** Before each model call the instructions get the user's briefing: where things stand, what is
  due or was promised, the rules that apply, what happened, then what is known that bears on their latest message,
  each line dated. `project=` keeps it to one project, `budget_chars=` caps it, and `briefing=False` reads only what
  bears on the question, as 0.1 did. An account without the briefing switched on gets that too, and the briefing is
  tried again ten minutes later. A briefing with nothing in it yet says so, so the model never reads silence as
  permission to guess.
- **`await memory.save(context, result, session=...)`** keeps the whole run, each tool call and its output included,
  in one memory for the conversation. The history an Agents SDK session hands the run is not sent again, a save that
  fails goes with the conversation's next one, and over 500 messages go 500 at a time. Hosted tools (web search,
  file search, code, a computer) are kept as the call, without what they brought back; reasoning is left out.
  `session=` on `GeniffyMemory` (a string or a function of the context) names the conversation for every save.
- `remember(context, message, result)` still saves one exchange, and takes `session=`.
- Needs `geniffy` 0.4.0 or later.

## 0.1.0

The first release.

- `GeniffyMemory(space=...)`: `run_config()` gives `Runner.run` a model input filter that puts what is known
  about the user in the instructions before each model call, after the agent's own. Geniffy is asked once per
  question however many steps a run takes, and a run goes on without memory when Geniffy can't be reached.
- `memory.remember(context, message, result)`: saves the exchange, from a string or input items and a
  `RunResult` or text.
- `geniffy_tools(space=...)`: `recall` and `remember` as function tools, reading the user from the run context.
- A space is required, and a blank one (or None from a function of the context) is refused, so one user's
  memory is never written to another's or to your own.
