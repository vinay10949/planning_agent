import asyncio
import os

from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from rich.console import Console

from planning_agent.agent import app as agent_app
from planning_agent.config import settings

console = Console()


def _init_langfuse() -> Langfuse:
    """Set Langfuse env vars and return a client (v2 SDK)."""
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


async def run_evaluations():
    langfuse = _init_langfuse()

    dataset_name = "planning-agent-eval"

    # Idempotent dataset creation — v2 returns the existing dataset if name is taken.
    langfuse.create_dataset(name=dataset_name)

    test_cases = [
        {
            "input": "What are the three tallest buildings in the world? Compare their heights.",
            "expected_topics": ["tallest", "building", "height"],
        },
        {
            "input": "Explain the water cycle in 3 steps, then give a real-world example.",
            "expected_topics": ["evaporation", "condensation", "precipitation"],
        },
        {
            "input": "Calculate how many seconds are in a year, then convert that to minutes.",
            "expected_topics": ["seconds", "year", "minutes"],
        },
    ]

    for tc in test_cases:
        langfuse.create_dataset_item(
            dataset_name=dataset_name,
            input={"question": tc["input"]},
            expected_output={"expected_topics": tc["expected_topics"]},
        )

    dataset_items = langfuse.get_dataset(dataset_name).items

    for item in dataset_items:
        console.print(f"\n🧪 Evaluating: [bold]{item.input['question'][:50]}...[/bold]")

        handler = CallbackHandler(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )

        result = await agent_app.ainvoke(
            {"input": item.input["question"], "plan": [], "past_steps": [], "response": ""},
            config={"callbacks": [handler]},
        )

        final_response = result.get("response", "")

        # v2 SDK exposes the most recent trace id via get_trace_id().
        trace_id = handler.get_trace_id()

        expected_topics = item.expected_output.get("expected_topics", [])
        covered = [t for t in expected_topics if t.lower() in final_response.lower()]
        score = len(covered) / len(expected_topics) if expected_topics else 0.0

        if trace_id:
            langfuse.score(
                trace_id=trace_id,
                name="topic_coverage",
                value=score,
                comment=f"Covered {covered} out of {expected_topics}",
            )

        handler.flush()
        langfuse.flush()

        console.print(f"   ✅ Topic Coverage: {score:.2f}")


if __name__ == "__main__":
    asyncio.run(run_evaluations())
