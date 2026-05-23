"""Unit tests for planning_agent.models."""

import operator
from typing import List, Tuple, Annotated

from pydantic import ValidationError
import pytest

from planning_agent.models import Plan, Replan, PlanExecuteState


# ─── Plan Model Tests ────────────────────────────────────────────────────────


class TestPlan:
    """Tests for the Plan Pydantic model."""

    def test_plan_creation_with_steps(self):
        steps = ["Step 1: Research", "Step 2: Analyze", "Step 3: Report"]
        plan = Plan(steps=steps)
        assert plan.steps == steps
        assert len(plan.steps) == 3

    def test_plan_creation_with_empty_steps(self):
        plan = Plan(steps=[])
        assert plan.steps == []
        assert len(plan.steps) == 0

    def test_plan_creation_with_single_step(self):
        plan = Plan(steps=["Do something"])
        assert plan.steps == ["Do something"]

    def test_plan_steps_field_description(self):
        """Verify the Field description is set correctly."""
        field_info = Plan.model_fields["steps"]
        assert field_info.description == "Different steps to follow, in order."

    def test_plan_model_schema(self):
        """Verify the JSON schema includes the expected structure."""
        schema = Plan.model_json_schema()
        assert "properties" in schema
        assert "steps" in schema["properties"]

    def test_plan_serialization(self):
        steps = ["Step A", "Step B"]
        plan = Plan(steps=steps)
        data = plan.model_dump()
        assert data == {"steps": ["Step A", "Step B"]}

    def test_plan_deserialization(self):
        data = {"steps": ["Step X", "Step Y", "Step Z"]}
        plan = Plan.model_validate(data)
        assert plan.steps == ["Step X", "Step Y", "Step Z"]

    def test_plan_steps_preserve_order(self):
        steps = ["First", "Second", "Third", "Fourth"]
        plan = Plan(steps=steps)
        assert plan.steps[0] == "First"
        assert plan.steps[3] == "Fourth"


# ─── Replan Model Tests ──────────────────────────────────────────────────────


class TestReplan:
    """Tests for the Replan Pydantic model."""

    def test_replan_with_remaining_steps(self):
        replan = Replan(
            feedback="Partially completed.",
            next_steps=["Continue with step 2", "Do step 3"],
        )
        assert replan.feedback == "Partially completed."
        assert replan.next_steps == ["Continue with step 2", "Do step 3"]

    def test_replan_task_complete_empty_next_steps(self):
        """When the task is fully complete, next_steps should be empty."""
        replan = Replan(
            feedback="All objectives have been met. The answer is 42.",
            next_steps=[],
        )
        assert replan.feedback == "All objectives have been met. The answer is 42."
        assert replan.next_steps == []

    def test_replan_feedback_field_description(self):
        field_info = Replan.model_fields["feedback"]
        assert "accomplished" in field_info.description.lower()

    def test_replan_next_steps_field_description(self):
        field_info = Replan.model_fields["next_steps"]
        assert "empty" in field_info.description.lower() or "remaining" in field_info.description.lower()

    def test_replan_serialization(self):
        replan = Replan(feedback="Done", next_steps=[])
        data = replan.model_dump()
        assert data == {"feedback": "Done", "next_steps": []}

    def test_replan_deserialization(self):
        data = {"feedback": "In progress", "next_steps": ["Step A"]}
        replan = Replan.model_validate(data)
        assert replan.feedback == "In progress"
        assert replan.next_steps == ["Step A"]


# ─── PlanExecuteState TypedDict Tests ────────────────────────────────────────


class TestPlanExecuteState:
    """Tests for the PlanExecuteState TypedDict."""

    def test_state_creation_minimal(self):
        state: PlanExecuteState = {
            "input": "What is the capital of France?",
            "plan": [],
            "past_steps": [],
            "response": "",
        }
        assert state["input"] == "What is the capital of France?"
        assert state["plan"] == []
        assert state["past_steps"] == []
        assert state["response"] == ""

    def test_state_creation_with_plan(self):
        state: PlanExecuteState = {
            "input": "Calculate 2+2",
            "plan": ["Add 2 and 2", "Report the result"],
            "past_steps": [],
            "response": "",
        }
        assert len(state["plan"]) == 2
        assert state["plan"][0] == "Add 2 and 2"

    def test_state_with_past_steps(self):
        state: PlanExecuteState = {
            "input": "Research topic",
            "plan": ["Write summary"],
            "past_steps": [("Search web", "Found results")],
            "response": "",
        }
        assert state["past_steps"][0] == ("Search web", "Found results")

    def test_state_with_response(self):
        state: PlanExecuteState = {
            "input": "Some query",
            "plan": [],
            "past_steps": [("Step 1", "Result 1")],
            "response": "The final answer is 42.",
        }
        assert state["response"] == "The final answer is 42."

    def test_state_all_fields_populated(self):
        state: PlanExecuteState = {
            "input": "Complex task",
            "plan": ["Step A", "Step B"],
            "past_steps": [("Step 0", "Done")],
            "response": "",
        }
        assert state["input"] == "Complex task"
        assert len(state["plan"]) == 2
        assert len(state["past_steps"]) == 1
        assert state["response"] == ""
