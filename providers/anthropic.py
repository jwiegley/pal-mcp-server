"""Anthropic (Claude) model provider implementation.

Talks directly to the Anthropic Messages API via the official ``anthropic``
Python SDK rather than routing Claude through OpenRouter or DIAL. This gives
PAL first-class access to native features the OpenAI-compatible shim does not
expose: extended/adaptive thinking, accurate token counting, and prompt caching.
"""

import logging
from typing import TYPE_CHECKING, ClassVar, Optional

if TYPE_CHECKING:
    from tools.models import ToolModelCategory  # noqa: F401  # used by category routing added in Task 5

from .base import ModelProvider
from .registries.anthropic import AnthropicModelRegistry
from .registry_provider_mixin import RegistryBackedProviderMixin
from .shared import ModelCapabilities, ModelResponse, ProviderType

logger = logging.getLogger(__name__)


class AnthropicModelProvider(RegistryBackedProviderMixin, ModelProvider):
    """First-party Anthropic integration built on the official Anthropic SDK."""

    FRIENDLY_NAME = "Anthropic"
    REGISTRY_CLASS = AnthropicModelRegistry
    MODEL_CAPABILITIES: ClassVar[dict[str, ModelCapabilities]] = {}

    DEFAULT_BASE_URL = "https://api.anthropic.com"

    # Canonical model identifiers used for tool-category routing.
    PRIMARY_MODEL = "claude-opus-4-8"
    FALLBACK_MODEL = "claude-sonnet-4-6"
    FAST_MODEL = "claude-haiku-4-5-20251001"

    # PAL thinking levels -> fraction of a model's max_thinking_tokens (budget scheme).
    THINKING_BUDGETS = {
        "minimal": 0.005,
        "low": 0.08,
        "medium": 0.33,
        "high": 0.67,
        "max": 1.0,
    }

    # PAL thinking levels -> Anthropic adaptive effort strings (adaptive scheme).
    THINKING_EFFORT = {
        "minimal": "low",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "max": "max",
    }

    # Anthropic requires budget_tokens >= 1024 when the budget scheme is used.
    MIN_THINKING_BUDGET = 1024

    def __init__(self, api_key: str, **kwargs):
        """Initialize the Anthropic provider with an API key and optional base URL."""
        self._ensure_registry()
        self._base_url = kwargs.pop("base_url", None) or self.DEFAULT_BASE_URL
        super().__init__(api_key, **kwargs)
        self._client = None
        self._invalidate_capability_cache()

    # ------------------------------------------------------------------
    # Client access
    # ------------------------------------------------------------------
    @property
    def client(self):
        """Lazily construct the Anthropic SDK client."""
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self.api_key, base_url=self._base_url)
        return self._client

    @property
    def base_url(self) -> str:
        return self._base_url

    def get_provider_type(self) -> ProviderType:
        return ProviderType.ANTHROPIC

    # ------------------------------------------------------------------
    # Request execution (implemented in Task 5)
    # ------------------------------------------------------------------
    def generate_content(
        self,
        prompt: str,
        model_name: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_output_tokens: Optional[int] = None,
        thinking_mode: str = "medium",
        images: Optional[list[str]] = None,
        **kwargs,
    ) -> ModelResponse:
        raise NotImplementedError("generate_content is implemented in Task 5")


# Load registry data at import time for registry consumers.
AnthropicModelProvider._ensure_registry()
