"""Checks around the model.

These filters are a cheap first line of defense, and pattern matching can always
be bypassed. The real protection is structural: the model has no tools, and only
`RecruiterCalendar` can create a booking.
"""

import re

from scheduling_agent.prompts import PROMPT_CANARY

MAX_MESSAGE_CHARS = 500
MAX_TURNS = 30

TURN_LIMIT_REPLY = "This chat has reached its limit. Please start a new one."
OFF_TOPIC_REPLY = "I can only help with booking a call or questions about the process."
UNSAFE_ANSWER_REPLY = (
    "I can't help with that here, but you can ask the recruiter during your call."
)

# The last alternative covers Portuguese, since candidates may write in it.
_INJECTION_PATTERN = re.compile(
    r"ignore\s+(all\s+|any\s+|the\s+|your\s+)*(previous|prior|above)"
    r"|(system|hidden|initial)\s+(prompt|instructions?)"
    r"|developer\s+mode"
    r"|you\s+are\s+now\b"
    r"|ignore\s+(as\s+|todas\s+as\s+)*instru[cç][oõ]es",
    re.IGNORECASE,
)

_BOOKING_CLAIM_PATTERN = re.compile(
    r"\bI(?:'ve| have)?\s+(?:booked|confirmed|reserved|scheduled)\b"
    r"|\b(?:meeting|interview|call|slot)\s+(?:is|has been)\s+"
    r"(?:booked|confirmed|reserved|scheduled)\b"
    r"|\byou(?:'re| are)\s+(?:booked|confirmed|all set)\b",
    re.IGNORECASE,
)

_CLOCK_TIME_PATTERN = re.compile(
    r"\b\d{1,2}:\d{2}\b|\b\d{1,2}\s?(?:am|pm)\b",
    re.IGNORECASE,
)

_EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def refusal_for(message: str, turns: int) -> str | None:
    """Return a refusal if the message must not reach the model, otherwise None."""
    if turns > MAX_TURNS:
        return TURN_LIMIT_REPLY
    if _INJECTION_PATTERN.search(message):
        return OFF_TOPIC_REPLY
    return None


def extract_email(message: str) -> str | None:
    """Read the email from the candidate's text, so the model cannot supply one."""
    match = _EMAIL_PATTERN.search(message)
    return match.group() if match else None


def is_safe_answer(answer: str) -> bool:
    """Reject answers that leak the prompt, or that state bookings or clock times.

    Answers come from the model, which knows neither. Only the calendar does.
    """
    return not (
        PROMPT_CANARY in answer
        or _BOOKING_CLAIM_PATTERN.search(answer)
        or _CLOCK_TIME_PATTERN.search(answer)
    )
