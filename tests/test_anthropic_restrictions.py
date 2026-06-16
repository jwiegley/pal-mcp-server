"""Tests for ANTHROPIC_ALLOWED_MODELS restriction wiring."""

import os
from unittest.mock import patch

from providers.anthropic import AnthropicModelProvider
from providers.shared import ProviderType
from utils.model_restrictions import ModelRestrictionService


def test_anthropic_env_var_registered():
    assert ModelRestrictionService.ENV_VARS[ProviderType.ANTHROPIC] == "ANTHROPIC_ALLOWED_MODELS"


@patch.dict(os.environ, {"ANTHROPIC_ALLOWED_MODELS": "claude-opus-4-8"})
def test_anthropic_restriction_loaded():
    service = ModelRestrictionService()
    assert service.is_allowed(ProviderType.ANTHROPIC, "claude-opus-4-8") is True
    assert service.is_allowed(ProviderType.ANTHROPIC, "claude-sonnet-4-6") is False


@patch.dict(os.environ, {"ANTHROPIC_ALLOWED_MODELS": "claude-opus-4-8"})
def test_provider_validate_model_name_honors_restrictions():
    """End-to-end: the provider's validate_model_name respects the allowlist (alias-aware)."""
    import utils.model_restrictions
    from providers.registry import ModelProviderRegistry

    utils.model_restrictions._restriction_service = None
    ModelProviderRegistry.reset_for_testing()

    provider = AnthropicModelProvider("test-key")
    assert provider.validate_model_name("claude-opus-4-8") is True
    assert provider.validate_model_name("opus-4.8") is True  # alias resolves to the allowed model
    assert provider.validate_model_name("claude-sonnet-4-6") is False
    assert provider.validate_model_name("sonnet-4.6") is False

    utils.model_restrictions._restriction_service = None
