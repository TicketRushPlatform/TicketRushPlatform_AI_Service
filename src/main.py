import os
import logging
from typing import Any, Dict

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from src.agent.auth_context import (
    InvalidAccessToken,
    reset_authenticated_user,
    scoped_thread_id,
    set_authenticated_user,
    verify_bearer_access_token,
)
from src.agent.graph import ainvoke_ticket_agent
from src.agent.services.http_service import (
    ServiceApiError,
    reset_downstream_authorization,
    set_downstream_authorization,
)
from src.song_recognition.songfinder_service import recognize_song_bytes

load_dotenv()


def resolve_log_level(value: str | None) -> int:
    raw_level = (value or "INFO").strip()
    if not raw_level:
        return logging.INFO
    if raw_level.isdigit():
        return int(raw_level)
    return logging.getLevelNamesMapping().get(raw_level.upper(), logging.INFO)


logging.basicConfig(level=resolve_log_level(os.getenv("LOG_LEVEL")))
logger = logging.getLogger(__name__)

app = FastAPI(title="SongFinder Backend")

JWT_SECRET = os.getenv("JWT_SECRET", "dev-only-secret")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    thread_id: str = Field(default="default", min_length=1)


class ChatResponse(BaseModel):
    message: str
    thread_id: str


def extract_agent_reply(agent_result: Any) -> str:
    messages = agent_result.get("messages") if isinstance(agent_result, dict) else None
    if not messages:
        return ""

    last_message = messages[-1]
    if isinstance(last_message, dict):
        return str(last_message.get("content") or "")

    return str(getattr(last_message, "content", "") or last_message)


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    authorization: str | None = Header(default=None),
) -> ChatResponse:
    logger.info(
        "chat request received thread_id=%s message=%r",
        request.thread_id,
        request.message,
    )
    scoped_id = scoped_thread_id(request.thread_id)
    user_token = None
    if authorization:
        try:
            user_token = set_authenticated_user(
                verify_bearer_access_token(authorization, secret=JWT_SECRET, algorithm=JWT_ALGORITHM)
            )
            scoped_id = scoped_thread_id(request.thread_id)
        except InvalidAccessToken as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    auth_token = set_downstream_authorization(authorization) if authorization else None
    try:
        agent_result = await ainvoke_ticket_agent(request.message, thread_id=scoped_id)
    except ServiceApiError as exc:
        logger.error("chat downstream service error thread_id=%s error=%s", request.thread_id, exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("chat agent request failed thread_id=%s", request.thread_id)
        raise HTTPException(status_code=502, detail=f"Chat agent request failed: {exc}") from exc
    finally:
        if auth_token is not None:
            reset_downstream_authorization(auth_token)
        if user_token is not None:
            reset_authenticated_user(user_token)

    try:
        reply = extract_agent_reply(agent_result)
        if not reply:
            logger.error("chat agent returned empty response thread_id=%s", request.thread_id)
            raise HTTPException(status_code=502, detail="Chat agent returned an empty response.")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("chat agent response handling failed thread_id=%s", request.thread_id)
        raise HTTPException(status_code=502, detail=f"Chat agent response handling failed: {exc}") from exc

    logger.info("chat agent reply thread_id=%s reply=%r", request.thread_id, reply)
    return ChatResponse(message=reply, thread_id=request.thread_id)


@app.post("/api/recognize/songfinder")
async def recognize_song(
    audio: UploadFile = File(...),
    start_time: int = Query(0, alias="startTime"),
) -> Dict[str, Any]:
    audio_bytes = await audio.read()

    if not audio_bytes:
        raise HTTPException(
            status_code=400,
            detail="Empty audio file",
        )

    try:
        return await recognize_song_bytes(audio_bytes, start_time=start_time)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail={
                "message": "SongFinder API error",
                "status_code": exc.response.status_code,
                "response": exc.response.text,
            },
        ) from exc
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="SongFinder request timeout",
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Cannot connect to SongFinder API: {str(e)}",
        )
