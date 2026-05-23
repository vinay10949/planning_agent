"""Unit tests for planning_agent.config."""

import os
import pytest
from unittest.mock import patch

from planning_agent.config import Settings


class TestSettings:
    """Tests for the Settings configuration model."""

    def test_default_lm_studio_base_url(self):
        s = Settings()
        assert s.lm_studio_base_url == "http://localhost:1234/v1"

    def test_default_lm_studio_api_key(self):
        s = Settings()
        assert s.lm_studio_api_key == "lm-studio"

    def test_default_lm_studio_model(self):
        s = Settings()
        assert s.lm_studio_model == "qwen2.5-7b-instruct"

    def test_default_langfuse_public_key(self):
        s = Settings()
        assert s.langfuse_public_key == ""

    def test_default_langfuse_secret_key(self):
        s = Settings()
        assert s.langfuse_secret_key == ""

    def test_default_langfuse_host(self):
        s = Settings()
        assert s.langfuse_host == "http://localhost:3000"

    def test_override_from_env_vars(self):
        with patch.dict(os.environ, {
            "LM_STUDIO_BASE_URL": "http://custom-host:9999/v1",
            "LM_STUDIO_MODEL": "llama-3.1-8b-instruct",
        }):
            s = Settings()
            assert s.lm_studio_base_url == "http://custom-host:9999/v1"
            assert s.lm_studio_model == "llama-3.1-8b-instruct"

    def test_override_langfuse_from_env_vars(self):
        with patch.dict(os.environ, {
            "LANGFUSE_PUBLIC_KEY": "pk-lf-test-key",
            "LANGFUSE_SECRET_KEY": "sk-lf-test-secret",
            "LANGFUSE_HOST": "http://custom-langfuse:4000",
        }):
            s = Settings()
            assert s.langfuse_public_key == "pk-lf-test-key"
            assert s.langfuse_secret_key == "sk-lf-test-secret"
            assert s.langfuse_host == "http://custom-langfuse:4000"

    def test_extra_env_vars_are_ignored(self):
        """The config uses extra='ignore', so unknown env vars shouldn't cause errors."""
        with patch.dict(os.environ, {
            "SOME_RANDOM_VAR": "should_not_matter",
        }):
            s = Settings()  # Should not raise
            assert s.lm_studio_base_url == "http://localhost:1234/v1"

    def test_model_config_env_file_encoding(self):
        assert Settings.model_config["env_file_encoding"] == "utf-8"

    def test_model_config_env_file(self):
        assert Settings.model_config["env_file"] == ".env"

    def test_model_config_extra_ignore(self):
        assert Settings.model_config["extra"] == "ignore"
