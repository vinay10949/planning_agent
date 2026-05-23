"""Unit tests for planning_agent.agent — tools, graph nodes, and routing logic."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from planning_agent.models import Plan, Replan, PlanExecuteState
from planning_agent.agent import (
    fake_web_search,
    calculator,
    tools,
    execute_step,
    plan_step,
    replan_step,
    should_end,
    app,
    get_llm,
)


# ─── Tool Tests ───────────────────────────────────────────────────────────────


class TestFakeWebSearch:
    """Tests for the fake_web_search tool."""

    def test_returns_string(self):
        result = fake_web_search.invoke({"query": "capital of France"})
        assert isinstance(result, str)

    def test_includes_query_in_result(self):
        result = fake_web_search.invoke({"query": "capital of France"})
        assert "capital of France" in result

    def test_contains_preset_knowledge(self):
        result = fake_web_search.invoke({"query": "anything"})
        assert "Paris" in result
        assert "Berlin" in result

    def test_tool_has_name(self):
        assert fake_web_search.name == "fake_web_search"

    def test_tool_has_description(self):
        assert fake_web_search.description  # not empty

    def test_tool_description_mentions_search(self):
        assert "search" in fake_web_search.description.lower()


class TestCalculator:
    """Tests for the calculator tool."""

    def test_simple_addition(self):
        assert calculator.invoke({"expression": "2 + 2"}) == "4"

    def test_multiplication(self):
        assert calculator.invoke({"expression": "3 * 7"}) == "21"

    def test_division(self):
        assert calculator.invoke({"expression": "10 / 2"}) == "5.0"

    def test_complex_expression(self):
        result = calculator.invoke({"expression": "60 * 60 * 24 * 365"})
        assert result == "31536000"

    def test_subtraction(self):
        assert calculator.invoke({"expression": "100 - 37"}) == "63"

    def test_floating_point(self):
        result = calculator.invoke({"expression": "1 / 3"})
        assert float(result) == pytest.approx(0.3333, rel=1e-3)

    def test_invalid_expression_returns_error(self):
        result = calculator.invoke({"expression": "not_valid_python!!!"})
        assert "Error" in result

    def test_tool_has_name(self):
        assert calculator.name == "calculator"

    def test_tool_has_description(self):
        assert calculator.description

    def test_tool_description_mentions_math(self):
        desc = calculator.description.lower()
        assert "math" in desc or "compute" in desc or "calcul" in desc


class TestToolsList:
    """Tests for the tools collection."""

    def test_two_tools_registered(self):
        assert len(tools) == 2

    def test_tools_contain_fake_web_search(self):
        names = [t.name for t in tools]
        assert "fake_web_search" in names

    def test_tools_contain_calculator(self):
        names = [t.name for t in tools]
        assert "calculator" in names


# ─── should_end Routing Tests ────────────────────────────────────────────────


class TestShouldEnd:
    """Tests for the conditional edge function should_end."""

    def test_returns_end_when_response_present(self):
        state: PlanExecuteState = {
            "input": "test",
            "plan": ["some step"],
            "past_steps": [],
            "response": "The task is complete.",
        }
        assert should_end(state) == "end"

    def test_returns_end_when_plan_empty(self):
        state: PlanExecuteState = {
            "input": "test",
            "plan": [],
            "past_steps": [],
            "response": "",
        }
        assert should_end(state) == "end"

    def test_returns_end_when_response_and_empty_plan(self):
        state: PlanExecuteState = {
            "input": "test",
            "plan": [],
            "past_steps": [],
            "response": "Done!",
        }
        assert should_end(state) == "end"

    def test_returns_continue_when_plan_exists_and_no_response(self):
        state: PlanExecuteState = {
            "input": "test",
            "plan": ["Step 1", "Step 2"],
            "past_steps": [],
            "response": "",
        }
        assert should_end(state) == "continue"

    def test_returns_continue_with_past_steps_but_remaining_plan(self):
        state: PlanExecuteState = {
            "input": "test",
            "plan": ["Step 2"],
            "past_steps": [("Step 1", "Result 1")],
            "response": "",
        }
        assert should_end(state) == "continue"


# ─── Graph Node Tests (with mocks) ───────────────────────────────────────────


class TestPlanStep:
    """Tests for the plan_step graph node."""

    @pytest.mark.asyncio
    async def test_plan_step_returns_plan(self):
        """plan_step should invoke the planner and return steps."""
        mock_plan = Plan(steps=["Step 1", "Step 2", "Step 3"])

        with patch("planning_agent.agent.planner") as mock_planner:
            mock_planner.ainvoke = AsyncMock(return_value=mock_plan)

            state: PlanExecuteState = {
                "input": "Do a complex task",
                "plan": [],
                "past_steps": [],
                "response": "",
            }
            result = await plan_step(state)

            assert "plan" in result
            assert result["plan"] == ["Step 1", "Step 2", "Step 3"]
            mock_planner.ainvoke.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_plan_step_passes_objective_to_planner(self):
        """plan_step should pass the input as objective."""
        mock_plan = Plan(steps=["A"])

        with patch("planning_agent.agent.planner") as mock_planner:
            mock_planner.ainvoke = AsyncMock(return_value=mock_plan)

            state: PlanExecuteState = {
                "input": "My specific objective",
                "plan": [],
                "past_steps": [],
                "response": "",
            }
            await plan_step(state)

            mock_planner.ainvoke.assert_awaited_once_with(
                {"objective": "My specific objective"}
            )

    @pytest.mark.asyncio
    async def test_plan_step_retries_on_failure(self):
        """plan_step should retry on failure and succeed on a later attempt."""
        mock_plan = Plan(steps=["Step 1"])

        with patch("planning_agent.agent.planner") as mock_planner:
            mock_planner.ainvoke = AsyncMock(
                side_effect=[Exception("JSON parse error"), mock_plan]
            )

            state: PlanExecuteState = {
                "input": "My task",
                "plan": [],
                "past_steps": [],
                "response": "",
            }
            result = await plan_step(state)

            assert result["plan"] == ["Step 1"]
            assert mock_planner.ainvoke.await_count == 2

    @pytest.mark.asyncio
    async def test_plan_step_fallback_to_single_step(self):
        """If all retries fail, plan_step should fall back to a single-step plan."""
        with patch("planning_agent.agent.planner") as mock_planner:
            mock_planner.ainvoke = AsyncMock(
                side_effect=Exception("JSON parse error")
            )

            state: PlanExecuteState = {
                "input": "My important task",
                "plan": [],
                "past_steps": [],
                "response": "",
            }
            result = await plan_step(state)

            # Fallback: the input itself becomes the single step
            assert result["plan"] == ["My important task"]


class TestExecuteStep:
    """Tests for the execute_step graph node."""

    @pytest.mark.asyncio
    async def test_execute_step_returns_past_steps_and_remaining_plan(self):
        """execute_step should record the executed step and remove it from plan."""
        mock_result = {
            "messages": [MagicMock(content="Step result: 2 + 2 = 4")]
        }

        with patch("planning_agent.agent.executor_agent") as mock_agent:
            mock_agent.ainvoke = AsyncMock(return_value=mock_result)

            state: PlanExecuteState = {
                "input": "Calculate something",
                "plan": ["Calculate 2+2", "Report result"],
                "past_steps": [],
                "response": "",
            }
            result = await execute_step(state)

            assert "past_steps" in result
            assert len(result["past_steps"]) == 1
            assert result["past_steps"][0][0] == "Calculate 2+2"
            assert result["past_steps"][0][1] == "Step result: 2 + 2 = 4"
            assert result["plan"] == ["Report result"]

    @pytest.mark.asyncio
    async def test_execute_step_removes_first_plan_item(self):
        """The executed step should be removed from the plan."""
        mock_result = {
            "messages": [MagicMock(content="Done")]
        }

        with patch("planning_agent.agent.executor_agent") as mock_agent:
            mock_agent.ainvoke = AsyncMock(return_value=mock_result)

            state: PlanExecuteState = {
                "input": "test",
                "plan": ["Step A", "Step B", "Step C"],
                "past_steps": [],
                "response": "",
            }
            result = await execute_step(state)

            assert result["plan"] == ["Step B", "Step C"]

    @pytest.mark.asyncio
    async def test_execute_step_passes_objective_and_step_to_agent(self):
        """The agent should receive both the overall objective and the specific step."""
        mock_result = {
            "messages": [MagicMock(content="ok")]
        }

        with patch("planning_agent.agent.executor_agent") as mock_agent:
            mock_agent.ainvoke = AsyncMock(return_value=mock_result)

            state: PlanExecuteState = {
                "input": "Big objective",
                "plan": ["Do thing"],
                "past_steps": [],
                "response": "",
            }
            await execute_step(state)

            call_args = mock_agent.ainvoke.call_args[0][0]
            assert "Big objective" in call_args["messages"][0][1]
            assert "Do thing" in call_args["messages"][0][1]


class TestReplanStep:
    """Tests for the replan_step graph node."""

    @pytest.mark.asyncio
    async def test_replan_returns_remaining_steps_when_incomplete(self):
        """If the replanner finds remaining steps, it should return them as the new plan."""
        mock_replan = Replan(
            feedback="Partially done, still need to finish.",
            next_steps=["Step 2", "Step 3"],
        )

        with patch("planning_agent.agent.replanner") as mock_replanner:
            mock_replanner.ainvoke = AsyncMock(return_value=mock_replan)

            state: PlanExecuteState = {
                "input": "Do task",
                "plan": [],
                "past_steps": [("Step 1", "Done")],
                "response": "",
            }
            result = await replan_step(state)

            assert "plan" in result
            assert result["plan"] == ["Step 2", "Step 3"]

    @pytest.mark.asyncio
    async def test_replan_returns_final_answer_when_complete(self):
        """If the replanner's next_steps is empty, it should return the final answer."""
        mock_replan = Replan(
            feedback="The answer is 42. Task is fully complete.",
            next_steps=[],
        )

        with patch("planning_agent.agent.replanner") as mock_replanner:
            mock_replanner.ainvoke = AsyncMock(return_value=mock_replan)

            state: PlanExecuteState = {
                "input": "Do task",
                "plan": [],
                "past_steps": [("Step 1", "Result 1")],
                "response": "",
            }
            result = await replan_step(state)

            assert result["response"] == "The answer is 42. Task is fully complete."
            assert result["plan"] == []

    @pytest.mark.asyncio
    async def test_replan_passes_correct_context(self):
        """The replanner should receive the objective, plan text, and past steps."""
        mock_replan = Replan(feedback="ok", next_steps=[])

        with patch("planning_agent.agent.replanner") as mock_replanner:
            mock_replanner.ainvoke = AsyncMock(return_value=mock_replan)

            state: PlanExecuteState = {
                "input": "My objective",
                "plan": ["Remaining step"],
                "past_steps": [("Done step", "Its result")],
                "response": "",
            }
            await replan_step(state)

            call_kwargs = mock_replanner.ainvoke.call_args[0][0]
            assert call_kwargs["objective"] == "My objective"
            assert "1. Remaining step" in call_kwargs["plan"]
            assert "Done step" in call_kwargs["past_steps"]
            assert "Its result" in call_kwargs["past_steps"]

    @pytest.mark.asyncio
    async def test_replan_retries_on_failure(self):
        """replan_step should retry on failure and succeed on a later attempt."""
        mock_replan = Replan(feedback="Done", next_steps=[])

        with patch("planning_agent.agent.replanner") as mock_replanner:
            mock_replanner.ainvoke = AsyncMock(
                side_effect=[Exception("JSON parse error"), mock_replan]
            )

            state: PlanExecuteState = {
                "input": "Do task",
                "plan": [],
                "past_steps": [("Step 1", "Result 1")],
                "response": "",
            }
            result = await replan_step(state)

            assert result["response"] == "Done"
            assert mock_replanner.ainvoke.await_count == 2

    @pytest.mark.asyncio
    async def test_replan_fallback_uses_last_result(self):
        """If all replanner retries fail, it should return the last past_step result."""
        with patch("planning_agent.agent.replanner") as mock_replanner:
            mock_replanner.ainvoke = AsyncMock(
                side_effect=Exception("JSON parse error")
            )

            state: PlanExecuteState = {
                "input": "Do task",
                "plan": [],
                "past_steps": [("Step 1", "The final answer is 42")],
                "response": "",
            }
            result = await replan_step(state)

            assert result["response"] == "The final answer is 42"
            assert result["plan"] == []


# ─── Graph Structure Tests ────────────────────────────────────────────────────


class TestGraphStructure:
    """Tests for the compiled LangGraph application structure."""

    def test_app_is_compiled(self):
        """The graph app should be a compiled graph, not a raw StateGraph."""
        assert app is not None
        # Compiled graphs have an invoke method
        assert hasattr(app, "ainvoke")
        assert hasattr(app, "invoke")

    def test_graph_has_expected_nodes(self):
        """The compiled graph should contain planner, executor, and replanner nodes."""
        # Access the node names from the compiled graph
        node_names = set(app.get_graph().nodes.keys())
        # The graph may add a __start__ and __end__ node
        assert "planner" in node_names
        assert "executor" in node_names
        assert "replanner" in node_names


# ─── get_llm Tests ────────────────────────────────────────────────────────────


class TestGetLLM:
    """Tests for the get_llm factory function."""

    def test_get_llm_returns_chat_openai(self):
        from langchain_openai import ChatOpenAI
        llm = get_llm()
        assert isinstance(llm, ChatOpenAI)

    def test_get_llm_uses_settings(self):
        llm = get_llm()
        assert llm.model_name == "qwen2.5-7b-instruct"
