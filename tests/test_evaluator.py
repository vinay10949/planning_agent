"""Unit tests for planning_agent.evaluator — evaluation logic."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestTopicCoverageCalculation:
    """Tests for the topic coverage scoring logic used in evaluations."""

    def test_full_coverage(self):
        """All expected topics present → score 1.0."""
        final_response = "The tallest building is the Burj Khalifa. Its height is 828m."
        expected_topics = ["tallest", "building", "height"]
        covered = [t for t in expected_topics if t.lower() in final_response.lower()]
        score = len(covered) / len(expected_topics)
        assert score == 1.0

    def test_partial_coverage(self):
        """Some topics present → fractional score."""
        final_response = "The water cycle involves evaporation and condensation."
        expected_topics = ["evaporation", "condensation", "precipitation"]
        covered = [t for t in expected_topics if t.lower() in final_response.lower()]
        score = len(covered) / len(expected_topics)
        assert score == pytest.approx(2.0 / 3.0, rel=1e-3)

    def test_zero_coverage(self):
        """No expected topics present → score 0.0."""
        final_response = "Something completely unrelated to the expected topics."
        expected_topics = ["evaporation", "condensation", "precipitation"]
        covered = [t for t in expected_topics if t.lower() in final_response.lower()]
        score = len(covered) / len(expected_topics)
        assert score == 0.0

    def test_case_insensitive_coverage(self):
        """Topic matching should be case-insensitive."""
        final_response = "The TALLEST BUILDING has great HEIGHT."
        expected_topics = ["tallest", "building", "height"]
        covered = [t for t in expected_topics if t.lower() in final_response.lower()]
        score = len(covered) / len(expected_topics)
        assert score == 1.0

    def test_empty_expected_topics(self):
        """If expected_topics is empty, score should be 0.0 (avoids division by zero)."""
        final_response = "Some response"
        expected_topics = []
        score = len([t for t in expected_topics if t.lower() in final_response.lower()]) / len(expected_topics) if expected_topics else 0.0
        assert score == 0.0


class TestEvaluatorIntegration:
    """Integration-level tests for the evaluator with mocked LangFuse and Agent."""

    @pytest.mark.asyncio
    async def test_run_evaluations_creates_dataset(self):
        """run_evaluations should create a dataset and populate it with test cases."""
        mock_langfuse = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.items = []
        mock_langfuse.create_dataset.return_value = mock_dataset
        mock_langfuse.get_dataset.return_value = mock_dataset

        # Create mock dataset items
        mock_item = MagicMock()
        mock_item.input = {"question": "Test question?"}
        mock_item.expected_output = {"expected_topics": ["test"]}
        mock_item.id = "item-1"
        mock_dataset.items = [mock_item]

        # Mock the CallbackHandler
        mock_handler = MagicMock()
        mock_handler.last_trace_id = "trace-123"

        with patch("planning_agent.evaluator._init_langfuse", return_value=mock_langfuse), \
             patch("planning_agent.evaluator.agent_app") as mock_agent, \
             patch("planning_agent.evaluator.CallbackHandler", return_value=mock_handler):
            mock_agent.ainvoke = AsyncMock(return_value={
                "input": "Test question?",
                "plan": [],
                "past_steps": [],
                "response": "Test answer with test keyword",
            })

            from planning_agent.evaluator import run_evaluations
            await run_evaluations()

            # Verify dataset was created
            mock_langfuse.create_dataset.assert_called_once_with(name="planning-agent-eval")

    @pytest.mark.asyncio
    async def test_run_evaluations_scores_traces(self):
        """run_evaluations should score each trace with topic_coverage via create_score."""
        mock_langfuse = MagicMock()
        mock_dataset = MagicMock()
        mock_langfuse.create_dataset.return_value = mock_dataset

        mock_item = MagicMock()
        mock_item.input = {"question": "What is the capital of France?"}
        mock_item.expected_output = {"expected_topics": ["paris", "capital"]}
        mock_item.id = "item-test"
        mock_dataset.items = [mock_item]
        mock_langfuse.get_dataset.return_value = mock_dataset

        # Mock the CallbackHandler with a trace_id
        mock_handler = MagicMock()
        mock_handler.last_trace_id = "trace-abc"

        with patch("planning_agent.evaluator._init_langfuse", return_value=mock_langfuse), \
             patch("planning_agent.evaluator.agent_app") as mock_agent, \
             patch("planning_agent.evaluator.CallbackHandler", return_value=mock_handler):
            mock_agent.ainvoke = AsyncMock(return_value={
                "input": "What is the capital of France?",
                "plan": [],
                "past_steps": [],
                "response": "The capital of France is Paris.",
            })

            from planning_agent.evaluator import run_evaluations
            await run_evaluations()

            # Verify create_score was called with the trace_id
            mock_langfuse.create_score.assert_called_once()
            score_call_kwargs = mock_langfuse.create_score.call_args[1]
            assert score_call_kwargs["name"] == "topic_coverage"
            assert score_call_kwargs["value"] == 1.0  # Both "paris" and "capital" in response
            assert score_call_kwargs["trace_id"] == "trace-abc"

    @pytest.mark.asyncio
    async def test_run_evaluations_handles_delete_existing_dataset_run(self):
        """run_evaluations should try to delete existing dataset run before creating."""
        mock_langfuse = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.items = []
        mock_langfuse.create_dataset.return_value = mock_dataset
        mock_langfuse.get_dataset.return_value = mock_dataset

        mock_handler = MagicMock()
        mock_handler.last_trace_id = None

        with patch("planning_agent.evaluator._init_langfuse", return_value=mock_langfuse), \
             patch("planning_agent.evaluator.agent_app") as mock_agent, \
             patch("planning_agent.evaluator.CallbackHandler", return_value=mock_handler):
            mock_agent.ainvoke = AsyncMock(return_value={
                "input": "test",
                "plan": [],
                "past_steps": [],
                "response": "done",
            })

            from planning_agent.evaluator import run_evaluations
            await run_evaluations()

            # Should have tried to delete the dataset run first
            mock_langfuse.delete_dataset_run.assert_called_once_with(
                dataset_name="planning-agent-eval", run_name="default"
            )

    @pytest.mark.asyncio
    async def test_run_evaluations_flushes_after_scoring(self):
        """run_evaluations should call langfuse.flush() to ensure data is sent."""
        mock_langfuse = MagicMock()
        mock_dataset = MagicMock()
        mock_langfuse.create_dataset.return_value = mock_dataset

        # Need at least one item so the loop runs and flush() is called
        mock_item = MagicMock()
        mock_item.input = {"question": "test?"}
        mock_item.expected_output = {"expected_topics": ["test"]}
        mock_item.id = "item-1"
        mock_dataset.items = [mock_item]
        mock_langfuse.get_dataset.return_value = mock_dataset

        mock_handler = MagicMock()
        mock_handler.last_trace_id = "trace-xyz"

        with patch("planning_agent.evaluator._init_langfuse", return_value=mock_langfuse), \
             patch("planning_agent.evaluator.agent_app") as mock_agent, \
             patch("planning_agent.evaluator.CallbackHandler", return_value=mock_handler):
            mock_agent.ainvoke = AsyncMock(return_value={
                "input": "test",
                "plan": [],
                "past_steps": [],
                "response": "test response",
            })

            from planning_agent.evaluator import run_evaluations
            await run_evaluations()

            # Verify flush was called (once per dataset item)
            mock_langfuse.flush.assert_called()
