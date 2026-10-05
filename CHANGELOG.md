# Changelog

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
