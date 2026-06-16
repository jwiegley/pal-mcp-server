"""Tests for the direct Anthropic provider."""

from unittest.mock import MagicMock

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


def _make_mock_message(text="Hello from Claude", stop_reason="end_turn", input_tokens=12, output_tokens=8):
    """Build a MagicMock shaped like an anthropic Message response."""
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text

    message = MagicMock()
    message.content = [text_block]
    message.stop_reason = stop_reason
    message.usage = MagicMock()
    message.usage.input_tokens = input_tokens
    message.usage.output_tokens = output_tokens
    return message


class TestAnthropicGenerateContent:
    def setup_method(self):
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None

    def teardown_method(self):
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None

    def _provider_with_mock_client(self):
        provider = AnthropicModelProvider("test-key")
        mock_client = MagicMock()
        provider._client = mock_client
        return provider, mock_client

    def test_generate_content_resolves_alias_and_builds_request(self):
        provider, mock_client = self._provider_with_mock_client()
        mock_client.messages.create.return_value = _make_mock_message()

        result = provider.generate_content(
            prompt="What is 2+2?",
            model_name="opus-4.8",
            system_prompt="You are terse.",
            temperature=0.5,
            max_output_tokens=256,
        )

        mock_client.messages.create.assert_called_once()
        kwargs = mock_client.messages.create.call_args[1]
        # Alias resolved to canonical id.
        assert kwargs["model"] == "claude-opus-4-8"
        # max_tokens is required and forwarded.
        assert kwargs["max_tokens"] == 256
        # System prompt is a top-level param, not a message.
        assert kwargs["system"] == "You are terse."
        # Single user message with a content-block list.
        assert kwargs["messages"][0]["role"] == "user"
        assert kwargs["messages"][0]["content"][0] == {"type": "text", "text": "What is 2+2?"}

        assert result.content == "Hello from Claude"
        assert result.model_name == "claude-opus-4-8"
        assert result.provider == ProviderType.ANTHROPIC
        assert result.usage["input_tokens"] == 12
        assert result.usage["output_tokens"] == 8
        assert result.usage["total_tokens"] == 20

    def test_generate_content_defaults_max_tokens_from_capabilities(self):
        provider, mock_client = self._provider_with_mock_client()
        mock_client.messages.create.return_value = _make_mock_message()

        provider.generate_content(prompt="hi", model_name="claude-haiku-4-5-20251001")
        kwargs = mock_client.messages.create.call_args[1]
        assert kwargs["max_tokens"] == 64000  # haiku max_output_tokens

    def test_adaptive_model_uses_adaptive_thinking(self):
        provider, mock_client = self._provider_with_mock_client()
        mock_client.messages.create.return_value = _make_mock_message()

        provider.generate_content(prompt="think", model_name="opus-4.8", thinking_mode="high")
        kwargs = mock_client.messages.create.call_args[1]
        # Adaptive thinking goes through extra_body; no legacy budget block.
        assert "thinking" not in kwargs
        assert kwargs["extra_body"]["thinking"] == {"type": "adaptive"}
        assert kwargs["extra_body"]["output_config"] == {"effort": "high"}
        # Extended thinking forces temperature == 1.0.
        assert kwargs["temperature"] == 1.0

    def test_budget_model_uses_budget_thinking(self):
        provider, mock_client = self._provider_with_mock_client()
        mock_client.messages.create.return_value = _make_mock_message()

        provider.generate_content(prompt="think", model_name="claude-haiku-4-5-20251001", thinking_mode="medium")
        kwargs = mock_client.messages.create.call_args[1]
        # Budget scheme uses the typed thinking param.
        assert kwargs["thinking"]["type"] == "enabled"
        # 24000 * 0.33 == 7920, above the 1024 minimum.
        assert kwargs["thinking"]["budget_tokens"] == 7920
        assert "extra_body" not in kwargs
        assert kwargs["temperature"] == 1.0

    def test_temperature_clamped_when_no_thinking(self):
        provider, mock_client = self._provider_with_mock_client()
        mock_client.messages.create.return_value = _make_mock_message()

        # thinking_mode that is not a known budget level disables thinking.
        provider.generate_content(
            prompt="hi", model_name="claude-haiku-4-5-20251001", temperature=1.7, thinking_mode="off"
        )
        kwargs = mock_client.messages.create.call_args[1]
        assert "thinking" not in kwargs
        assert "extra_body" not in kwargs
        assert kwargs["temperature"] == 1.0  # clamped from 1.7 to Anthropic max of 1.0

    def test_response_parsing_concatenates_text_blocks(self):
        provider, mock_client = self._provider_with_mock_client()
        b1 = MagicMock()
        b1.type = "text"
        b1.text = "foo "
        thinking_block = MagicMock()
        thinking_block.type = "thinking"
        thinking_block.thinking = "reasoning"
        b2 = MagicMock()
        b2.type = "text"
        b2.text = "bar"
        msg = MagicMock()
        msg.content = [thinking_block, b1, b2]
        msg.stop_reason = "end_turn"
        msg.usage = MagicMock()
        msg.usage.input_tokens = 1
        msg.usage.output_tokens = 2
        mock_client.messages.create.return_value = msg

        result = provider.generate_content(prompt="x", model_name="opus-4.8", thinking_mode="off")
        assert result.content == "foo bar"  # thinking block excluded from content
        assert result.metadata["has_thinking"] is True

    def test_count_tokens_uses_sdk(self):
        provider, mock_client = self._provider_with_mock_client()
        mock_client.messages.count_tokens.return_value = MagicMock(input_tokens=42)
        assert provider.count_tokens("some text", "opus-4.8") == 42
        kwargs = mock_client.messages.count_tokens.call_args[1]
        assert kwargs["model"] == "claude-opus-4-8"

    def test_count_tokens_falls_back_to_heuristic(self):
        provider, mock_client = self._provider_with_mock_client()
        mock_client.messages.count_tokens.side_effect = RuntimeError("no network")
        # Heuristic is len//4; "abcdefgh" -> 2.
        assert provider.count_tokens("abcdefgh", "opus-4.8") == 2

    def test_api_error_wrapped_in_runtime_error(self):
        provider, mock_client = self._provider_with_mock_client()
        mock_client.messages.create.side_effect = ValueError("bad request")
        with pytest.raises(RuntimeError, match="Anthropic API error"):
            provider.generate_content(prompt="x", model_name="opus-4.8", thinking_mode="off")
