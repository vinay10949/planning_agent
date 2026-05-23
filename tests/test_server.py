"""Unit tests for planning_agent.server — FastAPI endpoints."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from planning_agent.server import api, AgentRequest, AgentResponse


@pytest.fixture
def client():
    return TestClient(api)


# ─── API Model Tests ──────────────────────────────────────────────────────────


class TestAgentRequest:
    """Tests for the AgentRequest model."""

    def test_request_with_query_only(self):
        req = AgentRequest(query="Hello")
        assert req.query == "Hello"
        assert req.session_id == "default-session"

    def test_request_with_custom_session_id(self):
        req = AgentRequest(query="Test query", session_id="user-456")
        assert req.session_id == "user-456"

    def test_request_query_required(self):
        with pytest.raises(Exception):
            AgentRequest()  # query is required


class TestAgentResponse:
    """Tests for the AgentResponse model."""

    def test_response_creation(self):
        resp = AgentResponse(response="The answer is 42", past_steps=[("Step 1", "Result 1")])
        assert resp.response == "The answer is 42"
        assert resp.past_steps == [("Step 1", "Result 1")]

    def test_response_with_empty_steps(self):
        resp = AgentResponse(response="Done", past_steps=[])
        assert resp.past_steps == []


# ─── API Endpoint Tests ──────────────────────────────────────────────────────


class TestRunEndpoint:
    """Tests for the POST /run endpoint."""

    def test_run_endpoint_success(self, client):
        """Successful agent invocation should return 200 with expected shape."""
        mock_result = {
            "input": "What is 2+2?",
            "plan": [],
            "past_steps": [("Calculate 2+2", "4")],
            "response": "The answer is 4.",
        }

        with patch("planning_agent.server.agent_app") as mock_agent, \
             patch("planning_agent.server.CallbackHandler") as mock_handler_cls:
            mock_agent.ainvoke = AsyncMock(return_value=mock_result)
            mock_handler_cls.return_value = MagicMock()

            response = client.post(
                "/run",
                json={"query": "What is 2+2?", "session_id": "test-user"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["response"] == "The answer is 4."
            assert len(data["past_steps"]) == 1

    def test_run_endpoint_uses_default_session(self, client):
        """If session_id is not provided, it should default to 'default-session'."""
        mock_result = {
            "input": "test",
            "plan": [],
            "past_steps": [],
            "response": "Done",
        }

        with patch("planning_agent.server.agent_app") as mock_agent, \
             patch("planning_agent.server.CallbackHandler") as mock_handler_cls:
            mock_agent.ainvoke = AsyncMock(return_value=mock_result)
            mock_handler_cls.return_value = MagicMock()

            response = client.post("/run", json={"query": "test"})

            assert response.status_code == 200

    def test_run_endpoint_agent_error_returns_500(self, client):
        """If the agent raises an exception, the endpoint should return 500."""
        with patch("planning_agent.server.agent_app") as mock_agent, \
             patch("planning_agent.server.CallbackHandler") as mock_handler_cls:
            mock_agent.ainvoke = AsyncMock(side_effect=RuntimeError("LLM connection failed"))
            mock_handler_cls.return_value = MagicMock()

            response = client.post("/run", json={"query": "test"})

            assert response.status_code == 500
            assert "LLM connection failed" in response.json()["detail"]

    def test_run_endpoint_passes_correct_input_to_agent(self, client):
        """The agent should receive the query in the correct state format."""
        mock_result = {
            "input": "My query",
            "plan": [],
            "past_steps": [],
            "response": "Final answer",
        }

        with patch("planning_agent.server.agent_app") as mock_agent, \
             patch("planning_agent.server.CallbackHandler") as mock_handler_cls:
            mock_agent.ainvoke = AsyncMock(return_value=mock_result)
            mock_handler_cls.return_value = MagicMock()

            client.post("/run", json={"query": "My query", "session_id": "sess-1"})

            call_args = mock_agent.ainvoke.call_args
            state_arg = call_args[0][0]
            assert state_arg["input"] == "My query"
            assert state_arg["plan"] == []
            assert state_arg["past_steps"] == []
            assert state_arg["response"] == ""

    def test_run_endpoint_creates_langfuse_handler(self, client):
        """A LangFuse CallbackHandler should be created for each request."""
        mock_result = {
            "input": "test",
            "plan": [],
            "past_steps": [],
            "response": "ok",
        }

        with patch("planning_agent.server.agent_app") as mock_agent, \
             patch("planning_agent.server.CallbackHandler") as mock_handler_cls:
            mock_agent.ainvoke = AsyncMock(return_value=mock_result)
            mock_handler_cls.return_value = MagicMock()

            client.post("/run", json={"query": "test query", "session_id": "u1"})

            mock_handler_cls.assert_called_once()
            # Verify the new v4+ API: public_key and trace_context are passed
            call_kwargs = mock_handler_cls.call_args[1]
            assert "public_key" in call_kwargs
            assert "trace_context" in call_kwargs
            # trace_id must be a valid 32-char lowercase hex string
            trace_id = call_kwargs["trace_context"]["trace_id"]
            assert len(trace_id) == 32
            assert all(c in "0123456789abcdef" for c in trace_id)

    def test_run_endpoint_missing_query_returns_422(self, client):
        """POST /run without a query field should return 422 validation error."""
        response = client.post("/run", json={})
        assert response.status_code == 422

    def test_run_endpoint_returns_empty_past_steps(self, client):
        """When the agent returns no past_steps, the response should have an empty list."""
        mock_result = {
            "input": "test",
            "plan": [],
            "past_steps": [],
            "response": "Direct answer",
        }

        with patch("planning_agent.server.agent_app") as mock_agent, \
             patch("planning_agent.server.CallbackHandler") as mock_handler_cls:
            mock_agent.ainvoke = AsyncMock(return_value=mock_result)
            mock_handler_cls.return_value = MagicMock()

            response = client.post("/run", json={"query": "test"})
            assert response.status_code == 200
            assert response.json()["past_steps"] == []


class TestAPIMetadata:
    """Tests for the FastAPI app metadata."""

    def test_api_title(self):
        assert api.title == "Planning Agent API"

    def test_api_version(self):
        assert api.version == "0.1.0"
