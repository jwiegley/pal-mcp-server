"""Tests for the Anthropic capability registry / manifest."""

from providers.registries.anthropic import AnthropicModelRegistry
from providers.shared import ProviderType


def test_registry_loads_expected_models():
    registry = AnthropicModelRegistry()
    models = registry.list_models()
    assert "claude-opus-4-8" in models
    assert "claude-sonnet-4-6" in models
    assert "claude-haiku-4-5-20251001" in models


def test_registry_assigns_anthropic_provider():
    registry = AnthropicModelRegistry()
    caps = registry.resolve("claude-opus-4-8")
    assert caps is not None
    assert caps.provider == ProviderType.ANTHROPIC
    # The manifest sets an explicit friendly_name, which the registry preserves
    # (the friendly_prefix default is only applied when none is supplied).
    assert caps.friendly_name == "Anthropic (Claude Opus 4.8)"


def test_registry_resolves_version_aliases():
    registry = AnthropicModelRegistry()
    assert registry.resolve("opus-4.8").model_name == "claude-opus-4-8"
    assert registry.resolve("sonnet-4.6").model_name == "claude-sonnet-4-6"
    assert registry.resolve("haiku-4.5").model_name == "claude-haiku-4-5-20251001"


def test_adaptive_models_declare_reasoning_effort():
    """Adaptive-thinking models are tagged via default_reasoning_effort; budget models are not."""
    registry = AnthropicModelRegistry()
    assert registry.resolve("claude-opus-4-8").default_reasoning_effort == "high"
    assert registry.resolve("claude-haiku-4-5-20251001").default_reasoning_effort is None
    assert registry.resolve("claude-haiku-4-5-20251001").max_thinking_tokens > 0


def test_no_duplicate_aliases():
    # Construction performs duplicate-alias detection; a clean construct means no dupes.
    registry = AnthropicModelRegistry()
    aliases = registry.list_aliases()
    assert len(aliases) == len(set(aliases))
