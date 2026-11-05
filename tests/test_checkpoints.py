"""The checkpointer must restore listed types, and only those."""

from typing import Any

import pytest
from pydantic import BaseModel

from scheduling_agent.graph import create_checkpointer
from scheduling_agent.models import Booking, Intent
from tests.fakes import TUESDAY_2PM, schedule


class Unlisted(BaseModel):
    value: int = 1


def round_trip(value: Any) -> Any:
    serializer = create_checkpointer().serde
    return serializer.loads_typed(serializer.dumps_typed(value))


@pytest.mark.parametrize(
    "value",
    [
        Booking(slot=TUESDAY_2PM, candidate_name="Ana", candidate_email="a@example.com"),
        schedule(preferred_date=TUESDAY_2PM.date(), preferred_time=TUESDAY_2PM.time()),
        Intent.CONFIRM,
    ],
)
def test_listed_types_are_restored(value: Any) -> None:
    restored = round_trip(value)

    assert type(restored) is type(value)
    assert restored == value


def test_unlisted_classes_are_not_rebuilt() -> None:
    assert not isinstance(round_trip(Unlisted()), Unlisted)
