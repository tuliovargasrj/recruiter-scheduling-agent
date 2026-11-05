"""Exercises the real OpenAI client against a fake HTTP transport, with no API calls."""

import json
from datetime import date, time
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from pydantic import SecretStr

from scheduling_agent.config import Settings
from scheduling_agent.llm import MAX_OUTPUT_TOKENS, LLMUnavailableError, OpenAILLM
from scheduling_agent.models import Intent, Interpretation, Stage
from tests.fakes import MONDAY_MORNING


def completion(content: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


class FakeOpenAI:
    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        self._status_code = status_code
        self._payload = payload
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        return httpx.Response(self._status_code, json=self._payload)

    def llm(self) -> OpenAILLM:
        settings = Settings(openai_api_key=SecretStr("test-key"), _env_file=None)
        return OpenAILLM(settings, httpx.Client(transport=httpx.MockTransport(self)))


def interpret(llm: OpenAILLM, message: str = "Tuesday please") -> Interpretation:
    return llm.interpret(message, Stage.COLLECTING, MONDAY_MORNING, [])


def test_interpret_parses_structured_output() -> None:
    content = json.dumps(
        {
            "intent": "schedule",
            "candidate_name": None,
            "preferred_date": "2026-11-10",
            "preferred_time": "14:00:00",
        }
    )
    result = interpret(FakeOpenAI(200, completion(content)).llm())

    assert result.intent is Intent.SCHEDULE
    assert result.preferred_date == date(2026, 11, 10)


def test_interpret_drops_the_utc_offset_the_model_may_add() -> None:
    content = json.dumps({"intent": "schedule", "preferred_time": "14:00:00-03:00"})
    result = interpret(FakeOpenAI(200, completion(content)).llm())

    assert result.preferred_time == time(14, 0)
    assert result.preferred_time is not None
    assert result.preferred_time.tzinfo is None


def test_interpret_sends_a_schema_openai_accepts_in_strict_mode() -> None:
    fake = FakeOpenAI(200, completion('{"intent": "other"}'))
    interpret(fake.llm())
    request = fake.requests[0]
    schema = request["response_format"]["json_schema"]["schema"]

    assert request["model"] == "gpt-4o-mini"
    assert request["temperature"] == 0
    assert request["max_completion_tokens"] == MAX_OUTPUT_TOKENS
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    assert all("default" not in field for field in schema["properties"].values())


def test_interpret_wraps_candidate_text_as_data() -> None:
    fake = FakeOpenAI(200, completion('{"intent": "other"}'))
    interpret(fake.llm(), "my {message}")
    system, user = fake.requests[0]["messages"]

    assert "Monday, 2026-11-09 08:00" in system["content"]
    assert "<candidate_message>\nmy {message}\n</candidate_message>" in user["content"]


def test_interpret_includes_recent_history() -> None:
    fake = FakeOpenAI(200, completion('{"intent": "other"}'))
    history: list[AnyMessage] = [HumanMessage("first question"), AIMessage("first reply")]
    fake.llm().interpret("next", Stage.COLLECTING, MONDAY_MORNING, history)
    user = fake.requests[0]["messages"][1]["content"]

    assert "Human: first question\nAI: first reply" in user


def test_answer_returns_the_model_text() -> None:
    llm = FakeOpenAI(200, completion("A 30-minute call.")).llm()

    assert llm.answer("How long?") == "A 30-minute call."


@pytest.mark.parametrize(
    ("status_code", "payload"),
    [
        (401, {"error": {"message": "bad key"}}),
        (200, completion("this is not JSON")),
        (200, completion('{"intent": "not-a-real-intent"}')),
    ],
)
def test_interpret_failures_become_a_single_error(
    status_code: int,
    payload: dict[str, Any],
) -> None:
    with pytest.raises(LLMUnavailableError):
        interpret(FakeOpenAI(status_code, payload).llm())
