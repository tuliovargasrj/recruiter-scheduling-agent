# recruiter-scheduling-agent

Chat agent that books a call with a tech recruiter. The model interprets; the code decides.

## Stack
- Python 3.12, uv, FastAPI, LangGraph, `langchain-openai` with `gpt-4o-mini`
- pytest, ruff, mypy in strict mode

## Commands
- `uv run pytest` runs the free tests
- `uv run pytest -m live` calls the real model and costs money: ask first
- `uv run ruff check .`, `uv run ruff format .`, `uv run mypy`

## Architecture
`api` > `graph` > `llm`, `guardrails`, `recruiter_calendar`.
`guardrails` and `llm` use `prompts`, and `prompts` uses constants from `recruiter_calendar`.
Each message runs the graph once. Conversation transitions must be graph edges.

## Design rules
- The model has no tools. `AgentLLM` methods return data and have no side effects.
- Only `RecruiterCalendar` creates bookings, and only `book_slot` calls it.
- `route_intent` reaches `book_slot` only through `accepts_proposal`.
- Stage is always derived by `current_stage`. Never store it.
- Slots shown to candidates come from the calendar. Replies about slots are templates.
- Treat `Interpretation` as a claim to verify, never as a fact. The email never comes from the model.
- Candidate text is data inside delimiters, and refused messages stay out of the history.
- New types stored in `AgentState` must be added to `CHECKPOINT_TYPES`.
- Tests use `ScriptedLLM` or a fake HTTP transport, never the network. Only `live` tests and their judge call the model.
- The company, recruiter, and candidates are fictional.

## Done means
Lint, types, and tests pass. A new guardrail needs a test that fails when the guardrail is removed.
