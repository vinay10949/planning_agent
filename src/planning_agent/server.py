import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langfuse import Langfuse
from langfuse.callback import CallbackHandler

from planning_agent.agent import app as agent_app
from planning_agent.config import settings

api = FastAPI(title="Planning Agent API", version="0.1.0")


def _init_langfuse() -> Langfuse:
    """Set Langfuse env vars and return a client instance.

    Langfuse v2 SDK has no global registry — each CallbackHandler is constructed
    with explicit credentials, and the module-level Langfuse() client is used only
    for explicit flush() / score / dataset operations.
    """
    if settings.langfuse_public_key:
        os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
    if settings.langfuse_secret_key:
        os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
    if settings.langfuse_host:
        os.environ["LANGFUSE_HOST"] = settings.langfuse_host

    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )


_langfuse_client = _init_langfuse()


class AgentRequest(BaseModel):
    query: str
    session_id: str = "default-session"


class AgentResponse(BaseModel):
    response: str
    past_steps: list[tuple[str, str]]


@api.post("/run", response_model=AgentResponse)
async def run_agent(request: AgentRequest):
    try:
        # v2 CallbackHandler ingests via /api/public/ingestion (supported on v2 server).
        langfuse_handler = CallbackHandler(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
            session_id=request.session_id,
        )

        config = {
            "configurable": {"thread_id": request.session_id},
            "callbacks": [langfuse_handler],
        }

        result = await agent_app.ainvoke(
            {"input": request.query, "plan": [], "past_steps": [], "response": ""},
            config=config,
        )

        # Flush both the handler's queue and the module client so traces land
        # before this request returns.
        langfuse_handler.flush()
        _langfuse_client.flush()

        return AgentResponse(
            response=result.get("response", ""),
            past_steps=result.get("past_steps", []),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
