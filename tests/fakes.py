from collections.abc import Sequence
from datetime import datetime
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient
from langchain_core.messages import AnyMessage

from scheduling_agent.api import build_app
from scheduling_agent.llm import LLMUnavailableError
from scheduling_agent.models import Intent, Interpretation, Stage

MONDAY_MORNING = datetime(2026, 11, 9, 8, 0, tzinfo=ZoneInfo("America/Sao_Paulo"))
TUESDAY_2PM = MONDAY_MORNING.replace(day=10, hour=14)

CONFIRM = Interpretation(intent=Intent.CONFIRM)
DECLINE = Interpretation(intent=Intent.DECLINE)
QUESTION = Interpretation(intent=Intent.QUESTION)
OTHER = Interpretation(intent=Intent.OTHER)
LLM_DOWN = LLMUnavailableError()


def fixed_now() -> datetime:
    return MONDAY_MORNING


def schedule(**fields: object) -> Interpretation:
    return Interpretation.model_validate({"intent": Intent.SCHEDULE, **fields})


class ScriptedLLM:
    """Returns prepared responses in order and records how often it was called."""

    def __init__(
        self,
        interpretations: Sequence[Interpretation | Exception] = (),
        answers: Sequence[str] = (),
    ) -> None:
        self._interpretations = list(interpretations)
        self._answers = list(answers)
        self.interpret_calls = 0
        self.history_seen: list[str] = []

    def interpret(
        self,
        message: str,
        stage: Stage,
        now: datetime,
        history: Sequence[AnyMessage],
    ) -> Interpretation:
        self.interpret_calls += 1
        self.history_seen = [item.text for item in history]
        result = self._interpretations.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def answer(self, question: str) -> str:
        return self._answers.pop(0)


def new_client(llm: ScriptedLLM) -> TestClient:
    return TestClient(build_app(llm, fixed_now))


class Chat:
    """One browser session."""

    def __init__(self, client: TestClient) -> None:
        self._client = client
        self._session_id = str(uuid4())

    def post(self, message: str) -> Any:
        return self._client.post(
            "/chat",
            json={"session_id": self._session_id, "message": message},
        )

    def send(self, message: str = "hello") -> dict[str, Any]:
        response = self.post(message)
        assert response.status_code == 200, response.text
        body: dict[str, Any] = response.json()
        return body


def chat_with(llm: ScriptedLLM) -> Chat:
    return Chat(new_client(llm))
