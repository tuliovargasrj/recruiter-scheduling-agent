"""Paid end-to-end tests against the real model. Run with `pytest -m live`."""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from scheduling_agent.api import create_app
from scheduling_agent.config import Settings
from tests.fakes import Chat
from tests.judge import judge_consent


def has_api_key() -> bool:
    try:
        Settings()
    except ValidationError:
        return False
    return True


pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not has_api_key(), reason="OPENAI_API_KEY is not set"),
]


def live_chat() -> Chat:
    return Chat(TestClient(create_app()))


def test_books_a_call_from_details_spread_across_messages() -> None:
    chat = live_chat()
    chat.send("Hi, I'm Ana Lima, ana@example.com. Can we talk next Tuesday?")

    assert chat.send("2pm please")["stage"] == "confirming"

    booked = chat.send("Yes, book it")
    assert booked["stage"] == "booked"
    assert booked["booked_slot"]


@pytest.mark.parametrize(
    ("reply", "should_book"),
    [
        ("Yes, book it", True),
        ("Sim, pode marcar", True),
        ("Yes, but can we do it a bit earlier?", False),
        ("Yes, but not on Tuesday", False),
        ("Claro, mas pode ser mais tarde?", False),
    ],
)
def test_only_a_plain_yes_books(reply: str, should_book: bool) -> None:
    chat = live_chat()
    proposal = chat.send("Hi, I'm Ana Lima, ana@example.com. Tuesday at 2pm please.")
    assert proposal["stage"] == "confirming", proposal

    result = chat.send(reply)

    booked = result["booked_slot"] is not None
    assert booked is should_book, result
    if not booked:
        assert "14:00" not in result["reply"], result

    verdict = judge_consent(proposal["reply"], reply)
    assert verdict.accepted is booked, verdict
