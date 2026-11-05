from calendar import SATURDAY
from collections.abc import Callable
from datetime import date, datetime, time, timedelta
from threading import Lock

from scheduling_agent.models import Booking

CALL_MINUTES = 30
BOOKING_WINDOW_DAYS = 14
MINIMUM_NOTICE = timedelta(hours=1)
WORKING_HOURS = (9, 10, 11, 14, 15, 16)


class SlotUnavailableError(Exception):
    """The requested slot is not open, usually because someone else just booked it."""


class CandidateAlreadyBookedError(Exception):
    """This email already holds a booking."""


class RecruiterCalendar:
    """In-memory calendar and the only component allowed to create a booking."""

    def __init__(self, now: Callable[[], datetime]) -> None:
        self._now = now
        self._bookings: dict[datetime, Booking] = {}
        self._lock = Lock()

    def open_slots(
        self,
        on: date | None = None,
        at: time | None = None,
    ) -> list[datetime]:
        """Return open slots, earliest first, optionally by day and start time."""
        return [
            slot
            for slot in self._working_slots()
            if slot not in self._bookings
            and (on is None or slot.date() == on)
            and (at is None or slot.time() == at)
        ]

    def book(self, slot: datetime, candidate_name: str, candidate_email: str) -> Booking:
        """Reserve a slot. The lock stops two conversations from taking the same one."""
        with self._lock:
            if slot in self._bookings or slot not in self._working_slots():
                raise SlotUnavailableError(slot.isoformat())

            email = candidate_email.casefold()
            if any(
                booking.candidate_email.casefold() == email
                for booking in self._bookings.values()
            ):
                raise CandidateAlreadyBookedError

            booking = Booking(
                slot=slot,
                candidate_name=candidate_name,
                candidate_email=candidate_email,
            )
            self._bookings[slot] = booking
            return booking

    def _working_slots(self) -> list[datetime]:
        now = self._now()
        earliest = now + MINIMUM_NOTICE
        days = (now.date() + timedelta(days=d) for d in range(BOOKING_WINDOW_DAYS))
        slots = (
            datetime.combine(day, time(hour), tzinfo=now.tzinfo)
            for day in days
            if day.weekday() < SATURDAY
            for hour in WORKING_HOURS
        )
        return [slot for slot in slots if slot >= earliest]
