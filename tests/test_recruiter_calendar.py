from calendar import SATURDAY
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from time import sleep

import pytest

from scheduling_agent.recruiter_calendar import (
    BOOKING_WINDOW_DAYS,
    CandidateAlreadyBookedError,
    RecruiterCalendar,
    SlotUnavailableError,
)
from tests.fakes import MONDAY_MORNING, TUESDAY_2PM, fixed_now

EMAIL = "ana@example.com"


def test_offers_the_earliest_slots_first() -> None:
    slots = RecruiterCalendar(fixed_now).open_slots()

    assert [slot.hour for slot in slots[:3]] == [9, 10, 11]
    assert slots[0].date() == MONDAY_MORNING.date()


def test_respects_the_minimum_notice() -> None:
    one_thirty_pm = MONDAY_MORNING.replace(hour=13, minute=30)
    calendar = RecruiterCalendar(lambda: one_thirty_pm)

    assert calendar.open_slots()[0].hour == 15


def test_never_offers_weekends_or_slots_past_the_window() -> None:
    slots = RecruiterCalendar(fixed_now).open_slots()
    last_day = MONDAY_MORNING.date() + timedelta(days=BOOKING_WINDOW_DAYS - 1)

    assert all(slot.weekday() < SATURDAY for slot in slots)
    assert all(slot.date() <= last_day for slot in slots)


def test_filters_by_day_and_time() -> None:
    calendar = RecruiterCalendar(fixed_now)
    saturday = MONDAY_MORNING.replace(day=14).date()

    assert calendar.open_slots(on=TUESDAY_2PM.date(), at=TUESDAY_2PM.time()) == [
        TUESDAY_2PM
    ]
    assert calendar.open_slots(on=saturday) == []


def test_booked_slot_is_no_longer_offered_or_bookable() -> None:
    calendar = RecruiterCalendar(fixed_now)
    calendar.book(TUESDAY_2PM, "Ana", EMAIL)

    assert TUESDAY_2PM not in calendar.open_slots()
    with pytest.raises(SlotUnavailableError):
        calendar.book(TUESDAY_2PM, "Bia", "bia@example.com")


def test_rejects_a_slot_outside_working_hours() -> None:
    three_am = TUESDAY_2PM.replace(hour=3)

    with pytest.raises(SlotUnavailableError):
        RecruiterCalendar(fixed_now).book(three_am, "Ana", EMAIL)


def test_rejects_a_second_booking_for_the_same_email() -> None:
    calendar = RecruiterCalendar(fixed_now)
    calendar.book(TUESDAY_2PM, "Ana", EMAIL)

    with pytest.raises(CandidateAlreadyBookedError):
        calendar.book(TUESDAY_2PM.replace(hour=15), "Ana", EMAIL.upper())


def test_concurrent_bookings_produce_exactly_one_winner() -> None:
    def clock_that_yields() -> datetime:
        # Booking reads the clock between its check and its write. Sleeping there
        # lets other threads run, so this test fails if the lock is removed.
        sleep(0.001)
        return fixed_now()

    calendar = RecruiterCalendar(clock_that_yields)

    def try_booking(attempt: int) -> bool:
        try:
            calendar.book(TUESDAY_2PM, "Candidate", f"c{attempt}@example.com")
        except SlotUnavailableError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(try_booking, range(32)))

    assert sum(results) == 1
