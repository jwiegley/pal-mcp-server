"""Tests for Anthropic wiring into the provider registry."""

import os
from unittest.mock import patch

from providers.anthropic import AnthropicModelProvider
from providers.registry import ModelProviderRegistry
from providers.shared import ProviderType


def test_anthropic_in_priority_order_above_dial_and_openrouter():
    order = ModelProviderRegistry.PROVIDER_PRIORITY_ORDER
    assert ProviderType.ANTHROPIC in order
    assert order.index(ProviderType.ANTHROPIC) < order.index(ProviderType.DIAL)
    assert order.index(ProviderType.ANTHROPIC) < order.index(ProviderType.OPENROUTER)


@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"})
def test_api_key_lookup_and_provider_construction():
    ModelProviderRegistry.reset_for_testing()
    ModelProviderRegistry.register_provider(ProviderType.ANTHROPIC, AnthropicModelProvider)
    provider = ModelProviderRegistry.get_provider(ProviderType.ANTHROPIC)
    assert provider is not None
    assert provider.get_provider_type() == ProviderType.ANTHROPIC
    ModelProviderRegistry.reset_for_testing()
