from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import cast
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from scheduling_agent.config import Settings
from scheduling_agent.graph import (
    AgentState,
    build_graph,
    create_checkpointer,
    current_stage,
)
from scheduling_agent.guardrails import MAX_MESSAGE_CHARS
from scheduling_agent.llm import AgentLLM, LLMUnavailableError, OpenAILLM
from scheduling_agent.models import Stage
from scheduling_agent.recruiter_calendar import RecruiterCalendar

CHAT_PAGE = Path(__file__).parent / "static" / "index.html"


class ChatRequest(BaseModel):
    session_id: UUID = Field(default_factory=uuid4)
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


class ChatResponse(BaseModel):
    session_id: UUID
    reply: str
    stage: Stage
    booked_slot: datetime | None = None


def build_app(llm: AgentLLM, now: Callable[[], datetime]) -> FastAPI:
    graph = build_graph(llm, RecruiterCalendar(now), now, create_checkpointer())
    app = FastAPI(title="Recruiter Scheduling Agent")

    @app.get("/", include_in_schema=False)
    def chat_page() -> FileResponse:
        return FileResponse(CHAT_PAGE)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat")
    def chat(request: ChatRequest) -> ChatResponse:
        config: RunnableConfig = {"configurable": {"thread_id": str(request.session_id)}}

        try:
            # Only the end of a turn is read back, so inner steps are not saved.
            result = graph.invoke(
                {"user_message": request.message},
                config,
                durability="exit",
            )
        except LLMUnavailableError as error:
            detail = "The assistant is unavailable. Please try again."
            raise HTTPException(status_code=503, detail=detail) from error

        state = cast(AgentState, result)
        booking = state.get("booking")
        return ChatResponse(
            session_id=request.session_id,
            reply=state["reply"],
            stage=current_stage(state),
            booked_slot=booking.slot if booking else None,
        )

    return app


def create_app() -> FastAPI:
    """Build the production application from the environment."""
    settings = Settings()
    timezone = settings.timezone
    return build_app(OpenAILLM(settings), lambda: datetime.now(timezone))
