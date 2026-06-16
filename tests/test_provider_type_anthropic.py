"""Tests for the ANTHROPIC provider type member."""

from providers.shared import ProviderType


def test_anthropic_provider_type_exists():
    assert ProviderType.ANTHROPIC.value == "anthropic"


def test_anthropic_provider_type_is_unique():
    values = [member.value for member in ProviderType]
    assert values.count("anthropic") == 1
