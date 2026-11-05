"""The conversation state machine.

The model only interprets messages. This graph decides what happens next, and
replies about slots and bookings are built here, so the model cannot invent them.

Each message runs the graph once. The checkpointer keeps the state between
messages, and that state alone determines the conversation's stage.
"""

from collections.abc import Callable
from datetime import date, datetime, time
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph

from scheduling_agent.guardrails import (
    OFF_TOPIC_REPLY,
    UNSAFE_ANSWER_REPLY,
    extract_email,
    is_safe_answer,
    refusal_for,
)
from scheduling_agent.llm import AgentLLM
from scheduling_agent.models import Booking, Intent, Interpretation, Stage
from scheduling_agent.prompts import RECRUITER_NAME
from scheduling_agent.recruiter_calendar import (
    BOOKING_WINDOW_DAYS,
    CALL_MINUTES,
    CandidateAlreadyBookedError,
    RecruiterCalendar,
    SlotUnavailableError,
)

MAX_OFFERED_SLOTS = 3

# Checkpoints restore only these project types, instead of any importable class.
CHECKPOINT_TYPES = [Booking, Intent, Interpretation]


class AgentState(TypedDict, total=False):
    # Per turn: written during a turn and never read by a later one.
    user_message: str
    interpretation: Interpretation
    blocked: bool
    reply: str
    # Conversation: kept across turns.
    history: Annotated[list[AnyMessage], add_messages]
    turns: int
    candidate_name: str
    candidate_email: str
    preferred_date: date | None
    preferred_time: time | None
    proposed_slot: datetime | None
    booking: Booking


def current_stage(state: AgentState) -> Stage:
    """Derive the stage from the facts, so contradictory states cannot exist."""
    if state.get("booking"):
        return Stage.BOOKED
    if state.get("proposed_slot"):
        return Stage.CONFIRMING
    return Stage.COLLECTING


def missing_details(state: AgentState) -> list[str]:
    required = {
        "full name": state.get("candidate_name"),
        "email address": state.get("candidate_email"),
    }
    return [label for label, value in required.items() if not value]


def repeats_proposal(interpretation: Interpretation, slot: datetime) -> bool:
    return interpretation.preferred_date in (None, slot.date()) and (
        interpretation.preferred_time in (None, slot.time())
    )


def accepts_proposal(state: AgentState) -> bool:
    """A "yes" that also names another time is a new request, not an acceptance."""
    interpretation = state["interpretation"]
    slot = state.get("proposed_slot")
    if interpretation.intent is not Intent.CONFIRM or slot is None:
        return False
    return repeats_proposal(interpretation, slot)


def _format_slot(slot: datetime) -> str:
    return f"{slot:%A, %B} {slot.day} at {slot:%H:%M}"


def _format_options(slots: list[datetime]) -> str:
    return "\n".join(f"- {_format_slot(slot)}" for slot in slots)


def _next_step(state: AgentState) -> str:
    """Remind the candidate of the open step after a detour."""
    stage = current_stage(state)
    slot = state.get("proposed_slot")
    if slot:
        return f"\n\nShall I book {_format_slot(slot)}?"
    if stage is Stage.COLLECTING:
        return f"\n\nWould you like to book a call with {RECRUITER_NAME}?"
    return ""


class SchedulingAgent:
    """Graph nodes. Each one reads the state and returns only the fields it changes."""

    def __init__(
        self,
        llm: AgentLLM,
        calendar: RecruiterCalendar,
        now: Callable[[], datetime],
    ) -> None:
        self._llm = llm
        self._calendar = calendar
        self._now = now

    def check_input(self, state: AgentState) -> AgentState:
        turns = state.get("turns", 0) + 1
        refusal = refusal_for(state["user_message"], turns)
        return {
            "turns": turns,
            "blocked": refusal is not None,
            "reply": refusal or "",
        }

    def interpret(self, state: AgentState) -> AgentState:
        """Ask the model, and keep only details that can be trusted.

        Details are saved here, before routing, so a message that combines a
        question with a name or a time loses nothing.
        """
        message = state["user_message"]
        interpretation = self._llm.interpret(
            message,
            current_stage(state),
            self._now(),
            state.get("history", []),
        )

        # The model may copy the proposed slot from the conversation. That is not
        # a request for that time, so it must not outlive this turn.
        slot = state.get("proposed_slot")
        requested_time = (
            None
            if slot and repeats_proposal(interpretation, slot)
            else interpretation.preferred_time
        )

        update: AgentState = {
            "interpretation": interpretation,
            "preferred_date": interpretation.preferred_date
            or state.get("preferred_date"),
            "preferred_time": requested_time or state.get("preferred_time"),
        }
        name = (interpretation.candidate_name or "").strip()
        if name and name.casefold() in message.casefold():
            update["candidate_name"] = name
        email = extract_email(message)
        if email:
            update["candidate_email"] = email
        return update

    def ask_for_details(self, state: AgentState) -> AgentState:
        missing = " and ".join(missing_details(state))
        return {"reply": f"Happy to help. What is your {missing}?"}

    def propose_slot(self, state: AgentState) -> AgentState:
        """Propose a slot only if the calendar has it, whatever the model understood."""
        preferred_date = state.get("preferred_date")
        preferred_time = state.get("preferred_time")
        rejected = state.get("proposed_slot")
        matches = [
            slot
            for slot in self._calendar.open_slots(on=preferred_date, at=preferred_time)
            if slot != rejected
        ]

        # A time identifies one slot. A day alone is too broad, so its slots are listed.
        if matches and preferred_time:
            slot = matches[0]
            return {
                "proposed_slot": slot,
                "preferred_time": None,
                "reply": f"{RECRUITER_NAME} is free on {_format_slot(slot)}. Book it?",
            }
        if matches:
            # One day is short enough to show whole, so "later" can find a later slot.
            options = matches if preferred_date else matches[:MAX_OFFERED_SLOTS]
            return {
                "proposed_slot": None,
                "preferred_time": None,
                "reply": f"Here are the open slots:\n{_format_options(options)}",
            }
        return self._offer_open_slots("That isn't available.") | {"preferred_date": None}

    def book_slot(self, state: AgentState) -> AgentState:
        slot = state["proposed_slot"]
        assert slot is not None  # `accepts_proposal` is the only route to this node.

        try:
            booking = self._calendar.book(
                slot,
                state["candidate_name"],
                state["candidate_email"],
            )
        except SlotUnavailableError:
            return self._offer_open_slots("Sorry, that slot was just taken.")
        except CandidateAlreadyBookedError:
            return {
                "proposed_slot": None,
                "reply": (
                    "This email already has a call booked. "
                    f"To change it, please contact {RECRUITER_NAME} directly."
                ),
            }

        return {
            "booking": booking,
            "proposed_slot": None,
            "reply": (
                f"You're booked with {RECRUITER_NAME} on {_format_slot(slot)} "
                f"for a {CALL_MINUTES}-minute video call."
            ),
        }

    def decline_slot(self, state: AgentState) -> AgentState:
        return {
            "proposed_slot": None,
            "preferred_date": None,
            "preferred_time": None,
            "reply": "No problem. Tell me another day or time if you'd like a call.",
        }

    def answer_question(self, state: AgentState) -> AgentState:
        return {"reply": self._llm.answer(state["user_message"])}

    def check_answer(self, state: AgentState) -> AgentState:
        answer = state["reply"]
        safe_answer = answer if is_safe_answer(answer) else UNSAFE_ANSWER_REPLY
        return {"reply": f"{safe_answer}{_next_step(state)}"}

    def redirect(self, state: AgentState) -> AgentState:
        return {"reply": f"{OFF_TOPIC_REPLY}{_next_step(state)}"}

    def already_booked(self, state: AgentState) -> AgentState:
        slot = state["booking"].slot
        return {
            "reply": (
                f"You already have a call on {_format_slot(slot)}. "
                f"To change it, please contact {RECRUITER_NAME} directly."
            )
        }

    def record_turn(self, state: AgentState) -> AgentState:
        return {
            "history": [HumanMessage(state["user_message"]), AIMessage(state["reply"])]
        }

    def _offer_open_slots(self, intro: str) -> AgentState:
        slots = self._calendar.open_slots()
        options = (
            f"The closest open slots are:\n{_format_options(slots[:MAX_OFFERED_SLOTS])}"
            if slots
            else f"There are no open slots in the next {BOOKING_WINDOW_DAYS} days."
        )
        return {
            "proposed_slot": None,
            "preferred_time": None,
            "reply": f"{intro} {options}",
        }


def route_after_input(state: AgentState) -> Literal["__end__", "interpret"]:
    """Refused turns end here, which keeps them out of the history the model reads."""
    return "__end__" if state["blocked"] else "interpret"


def route_intent(
    state: AgentState,
) -> Literal[
    "answer_question",
    "already_booked",
    "redirect",
    "decline_slot",
    "book_slot",
    "ask_for_details",
    "propose_slot",
]:
    """Choose the next node. These rules, not the model, control the conversation."""
    intent = state["interpretation"].intent

    if intent is Intent.QUESTION:
        return "answer_question"
    if current_stage(state) is Stage.BOOKED:
        return "already_booked"
    if intent is Intent.OTHER:
        return "redirect"
    if intent is Intent.DECLINE:
        return "decline_slot"
    if accepts_proposal(state):
        return "book_slot"
    # Scheduling details, a new time, or a "yes" with nothing to accept.
    return "ask_for_details" if missing_details(state) else "propose_slot"


def create_checkpointer() -> InMemorySaver:
    serializer = JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)
    return InMemorySaver(serde=serializer)


def build_graph(
    llm: AgentLLM,
    calendar: RecruiterCalendar,
    now: Callable[[], datetime],
    checkpointer: BaseCheckpointSaver[str],
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    agent = SchedulingAgent(llm, calendar, now)
    graph = StateGraph(AgentState)

    for node in (
        agent.check_input,
        agent.interpret,
        agent.ask_for_details,
        agent.propose_slot,
        agent.book_slot,
        agent.decline_slot,
        agent.answer_question,
        agent.check_answer,
        agent.redirect,
        agent.already_booked,
        agent.record_turn,
    ):
        graph.add_node(node)

    graph.add_edge(START, "check_input")
    graph.add_conditional_edges("check_input", route_after_input)
    graph.add_conditional_edges("interpret", route_intent)
    graph.add_edge("answer_question", "check_answer")

    for node_name in (
        "ask_for_details",
        "propose_slot",
        "book_slot",
        "decline_slot",
        "check_answer",
        "redirect",
        "already_booked",
    ):
        graph.add_edge(node_name, "record_turn")
    graph.add_edge("record_turn", END)

    return graph.compile(checkpointer=checkpointer)
