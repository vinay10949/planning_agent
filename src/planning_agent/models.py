from pydantic import BaseModel, Field
from typing import List, Tuple, Annotated
from typing_extensions import TypedDict
import operator

class Plan(BaseModel):
    """List of steps to accomplish the user's task."""
    steps: List[str] = Field(description="Different steps to follow, in order.")

class Replan(BaseModel):
    """Revised plan after executing a step, or signal completion."""
    feedback: str = Field(description="Analysis of what was accomplished and what still needs to be done.")
    next_steps: List[str] = Field(description="Remaining steps. Empty list if the task is fully complete.")

class PlanExecuteState(TypedDict):
    """State that flows through the Planning Agent graph."""
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple[str, str]], operator.add]
    response: str
