"""Tests for the direct Anthropic provider."""

import os  # noqa: F401  # used by generate_content tests added in Task 5
from unittest.mock import MagicMock, patch  # noqa: F401  # used by Task 5 tests

import pytest

from providers.anthropic import AnthropicModelProvider
from providers.shared import ProviderType


class TestAnthropicProviderIdentity:
    def setup_method(self):
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None

    def teardown_method(self):
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None

    def test_initialization(self):
        provider = AnthropicModelProvider("test-key")
        assert provider.api_key == "test-key"
        assert provider.get_provider_type() == ProviderType.ANTHROPIC
        assert provider.base_url == "https://api.anthropic.com"

    def test_initialization_with_custom_url(self):
        provider = AnthropicModelProvider("test-key", base_url="https://proxy.example.com")
        assert provider.base_url == "https://proxy.example.com"

    def test_friendly_name(self):
        provider = AnthropicModelProvider("test-key")
        assert provider.FRIENDLY_NAME == "Anthropic"

    def test_model_validation(self):
        provider = AnthropicModelProvider("test-key")
        assert provider.validate_model_name("claude-opus-4-8") is True
        assert provider.validate_model_name("opus-4.8") is True
        assert provider.validate_model_name("sonnet-4.6") is True
        assert provider.validate_model_name("haiku-4.5") is True
        assert provider.validate_model_name("gpt-4") is False
        assert provider.validate_model_name("grok-4") is False

    def test_resolve_model_name(self):
        provider = AnthropicModelProvider("test-key")
        assert provider._resolve_model_name("opus-4.8") == "claude-opus-4-8"
        assert provider._resolve_model_name("claude-opus-4.8") == "claude-opus-4-8"
        assert provider._resolve_model_name("haiku-4.5") == "claude-haiku-4-5-20251001"
        assert provider._resolve_model_name("claude-opus-4-8") == "claude-opus-4-8"

    def test_get_capabilities(self):
        provider = AnthropicModelProvider("test-key")
        caps = provider.get_capabilities("opus-4.8")
        assert caps.model_name == "claude-opus-4-8"
        assert caps.provider == ProviderType.ANTHROPIC
        assert caps.context_window == 1_000_000
        assert caps.max_output_tokens == 128_000
        assert caps.supports_extended_thinking is True
        assert caps.supports_images is True
        assert caps.default_reasoning_effort == "high"

    def test_unsupported_model_capabilities(self):
        provider = AnthropicModelProvider("test-key")
        with pytest.raises(ValueError, match="Unsupported model 'invalid-model' for provider anthropic"):
            provider.get_capabilities("invalid-model")
