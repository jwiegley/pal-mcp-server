"""Enumeration describing which backend owns a given model."""

from enum import Enum

__all__ = ["ProviderType"]


class ProviderType(Enum):
    """Canonical identifiers for every supported provider backend."""

    GOOGLE = "google"
    OPENAI = "openai"
    AZURE = "azure"
    ANTHROPIC = "anthropic"
    XAI = "xai"
    FACTORY = "factory"
    OPENROUTER = "openrouter"
    CUSTOM = "custom"
    DIAL = "dial"
