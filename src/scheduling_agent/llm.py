from collections.abc import Sequence
from datetime import datetime
from typing import Protocol, cast

import httpx
import openai
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
    get_buffer_string,
)
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from scheduling_agent.config import Settings
from scheduling_agent.models import Interpretation, Stage
from scheduling_agent.prompts import (
    ANSWER_SYSTEM_PROMPT,
    ANSWER_USER_PROMPT,
    INTERPRET_SYSTEM_PROMPT,
    INTERPRET_USER_PROMPT,
)

HISTORY_MESSAGES = 6
MAX_OUTPUT_TOKENS = 300
REQUEST_TIMEOUT_SECONDS = 20
MAX_RETRIES = 2


class LLMUnavailableError(Exception):
    """The model could not produce a usable response."""


class AgentLLM(Protocol):
    """The only two things the agent asks of a model. Neither has side effects."""

    def interpret(
        self,
        message: str,
        stage: Stage,
        now: datetime,
        history: Sequence[AnyMessage],
    ) -> Interpretation: ...

    def answer(self, question: str) -> str: ...


class OpenAILLM:
    def __init__(self, settings: Settings, http_client: httpx.Client | None = None):
        model = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0,
            max_tokens=MAX_OUTPUT_TOKENS,
            timeout=REQUEST_TIMEOUT_SECONDS,
            max_retries=MAX_RETRIES,
            http_client=http_client,
        )
        self._model = model
        self._interpreter = model.with_structured_output(Interpretation)

    def interpret(
        self,
        message: str,
        stage: Stage,
        now: datetime,
        history: Sequence[AnyMessage],
    ) -> Interpretation:
        messages = [
            SystemMessage(
                INTERPRET_SYSTEM_PROMPT.format(
                    now=now.strftime("%A, %Y-%m-%d %H:%M %Z"),
                    stage=stage,
                )
            ),
            HumanMessage(
                INTERPRET_USER_PROMPT.format(
                    history=get_buffer_string(history[-HISTORY_MESSAGES:]),
                    message=message,
                )
            ),
        ]
        try:
            result = self._interpreter.invoke(messages)
        except (openai.OpenAIError, OutputParserException, ValidationError) as error:
            raise LLMUnavailableError from error
        return cast(Interpretation, result)

    def answer(self, question: str) -> str:
        messages = [
            SystemMessage(ANSWER_SYSTEM_PROMPT),
            HumanMessage(ANSWER_USER_PROMPT.format(question=question)),
        ]
        try:
            return self._model.invoke(messages).text
        except openai.OpenAIError as error:
            raise LLMUnavailableError from error
