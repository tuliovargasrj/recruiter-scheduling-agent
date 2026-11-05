from datetime import date, datetime, time
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class Stage(StrEnum):
    """Where a conversation is in the booking flow."""

    COLLECTING = "collecting"
    CONFIRMING = "confirming"
    BOOKED = "booked"


class Intent(StrEnum):
    QUESTION = "question"
    SCHEDULE = "schedule"
    CONFIRM = "confirm"
    DECLINE = "decline"
    OTHER = "other"


class Interpretation(BaseModel):
    """What the model understood from one message. Treated as a claim, not a fact."""

    intent: Intent = Field(
        description=(
            "question: asks about the recruiter, the call, or the hiring process. "
            "schedule: wants a call, gives a name, email, day, or time, or asks to "
            "change the proposed slot. "
            "confirm: accepts the slot just proposed exactly as it is. "
            "decline: rejects the proposed slot or stops scheduling. "
            "other: anything unrelated."
        )
    )
    candidate_name: str | None = Field(
        default=None, description="Full name, only if the candidate stated it."
    )
    preferred_date: date | None = Field(
        default=None, description="Requested day as an absolute date."
    )
    preferred_time: time | None = Field(
        default=None,
        description="Requested start time in the candidate's local time, unconverted.",
    )

    @field_validator("preferred_time")
    @classmethod
    def drop_utc_offset(cls, value: time | None) -> time | None:
        """Slots are wall-clock times, but the model may add an offset to the time."""
        return value.replace(tzinfo=None) if value else None


class Booking(BaseModel):
    slot: datetime
    candidate_name: str
    candidate_email: str
