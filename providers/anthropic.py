"""Anthropic (Claude) model provider implementation.

Talks directly to the Anthropic Messages API via the official ``anthropic``
Python SDK rather than routing Claude through OpenRouter or DIAL. This gives
PAL first-class access to native features the OpenAI-compatible shim does not
expose: extended/adaptive thinking, accurate token counting, and prompt caching.
"""

import base64
import logging
from typing import TYPE_CHECKING, ClassVar, Optional

from utils.image_utils import validate_image

if TYPE_CHECKING:
    from tools.models import ToolModelCategory

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
    # Request execution
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
        """Generate content using the Anthropic Messages API."""
        self.validate_parameters(model_name, temperature)
        capabilities = self.get_capabilities(model_name)
        resolved_model_name = self._resolve_model_name(model_name)

        # Build the user content blocks (text + optional images).
        content_blocks: list[dict] = [{"type": "text", "text": prompt}]
        if images and capabilities.supports_images:
            for image_path in images:
                block = self._process_image(image_path)
                if block:
                    content_blocks.append(block)
        elif images and not capabilities.supports_images:
            logger.warning(
                "Model %s does not support images, ignoring %d image(s)",
                resolved_model_name,
                len(images),
            )

        # Anthropic requires max_tokens; fall back to the model's documented ceiling.
        effective_max_tokens = max_output_tokens or capabilities.max_output_tokens or 4096

        thinking_params = self._resolve_thinking_params(capabilities, thinking_mode)

        request_kwargs: dict = {
            "model": resolved_model_name,
            "max_tokens": effective_max_tokens,
            "messages": [{"role": "user", "content": content_blocks}],
        }
        if system_prompt:
            request_kwargs["system"] = system_prompt

        # Adaptive-thinking models (those advertising default_reasoning_effort) reject the
        # temperature parameter entirely; only budget-scheme/non-thinking models accept it.
        is_adaptive = bool(capabilities.default_reasoning_effort)
        if thinking_params:
            request_kwargs.update(thinking_params)
            if not is_adaptive:
                # Budget-scheme extended thinking requires temperature == 1.0.
                request_kwargs["temperature"] = 1.0
        elif capabilities.supports_temperature and not is_adaptive:
            # Anthropic accepts temperature in [0, 1]; clamp to be safe.
            request_kwargs["temperature"] = min(max(temperature, 0.0), 1.0)

        max_retries = 4
        retry_delays = [1, 3, 5, 8]
        attempt_counter = {"value": 0}
        active_thinking_mode = thinking_mode if thinking_params else None

        def _attempt() -> ModelResponse:
            attempt_counter["value"] += 1
            response = self._create_message(request_kwargs)
            return self._build_model_response(response, resolved_model_name, active_thinking_mode)

        try:
            return self._run_with_retries(
                operation=_attempt,
                max_attempts=max_retries,
                delays=retry_delays,
                log_prefix=f"Anthropic API ({resolved_model_name})",
            )
        except Exception as exc:
            attempts = max(attempt_counter["value"], 1)
            raise RuntimeError(
                f"Anthropic API error for model {resolved_model_name} after {attempts} attempt"
                f"{'s' if attempts > 1 else ''}: {exc}"
            ) from exc

    def count_tokens(self, text: str, model_name: str) -> int:
        """Count tokens via the Anthropic SDK, falling back to the heuristic."""
        if not text:
            return 0
        resolved_model_name = self._resolve_model_name(model_name)
        try:
            result = self.client.messages.count_tokens(
                model=resolved_model_name,
                messages=[{"role": "user", "content": text}],
            )
            return int(result.input_tokens)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully to heuristic
            logger.debug("Anthropic token counting failed (%s); using heuristic", exc)
            return super().count_tokens(text, model_name)

    def close(self) -> None:
        """Close the underlying SDK client if it was created."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001 - best-effort cleanup at shutdown
                pass
            self._client = None

    def get_preferred_model(self, category: "ToolModelCategory", allowed_models: list[str]) -> Optional[str]:
        """Pick Anthropic's preferred model for a tool category from the allowed set."""
        from tools.models import ToolModelCategory

        if not allowed_models:
            return None

        def find_first(preferences: list[str]) -> Optional[str]:
            for model in preferences:
                if model in allowed_models:
                    return model
            return None

        if category == ToolModelCategory.FAST_RESPONSE:
            return find_first([self.FAST_MODEL, self.FALLBACK_MODEL, self.PRIMARY_MODEL]) or allowed_models[0]
        if category == ToolModelCategory.EXTENDED_REASONING:
            return (
                find_first([self.PRIMARY_MODEL, "claude-opus-4-7", "claude-opus-4-6", self.FALLBACK_MODEL])
                or allowed_models[0]
            )
        return find_first([self.PRIMARY_MODEL, self.FALLBACK_MODEL, self.FAST_MODEL]) or allowed_models[0]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_thinking_params(self, capabilities: ModelCapabilities, thinking_mode: str) -> dict:
        """Return the request kwargs that enable thinking for this model, or {} if none.

        Adaptive-thinking models (Opus 4.6+/Sonnet 4.6/Fable) advertise a
        ``default_reasoning_effort`` and use the adaptive scheme via ``extra_body``.
        Budget-scheme models (Haiku 4.5, Sonnet 4.5, Opus 4.5) use the typed
        ``thinking={"type": "enabled", "budget_tokens": N}`` param.
        """
        if not capabilities.supports_extended_thinking:
            return {}
        if thinking_mode not in self.THINKING_BUDGETS:
            return {}

        if capabilities.default_reasoning_effort:
            effort = self.THINKING_EFFORT.get(thinking_mode, "medium")
            return {
                "extra_body": {
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": effort},
                }
            }

        if capabilities.max_thinking_tokens > 0:
            budget = int(capabilities.max_thinking_tokens * self.THINKING_BUDGETS[thinking_mode])
            budget = max(budget, self.MIN_THINKING_BUDGET)
            return {"thinking": {"type": "enabled", "budget_tokens": budget}}

        return {}

    def _create_message(self, request_kwargs: dict):
        """Execute a Messages request via the streaming API and return the final Message.

        We always stream rather than calling ``messages.create()`` directly. The SDK
        refuses a non-streaming request *before any network call* when it estimates the
        request could exceed the ~10 minute idle-connection limit (high ``max_tokens`` or
        long input) — which is exactly what rejected PAL consensus on Opus 4.8, since the
        request falls back to the model's full ``max_output_tokens`` ceiling (128K).
        Streaming sidesteps that guard and provides timeout protection; ``get_final_message()``
        reconstructs the same Message object ``messages.create()`` would have returned, so
        downstream parsing is unchanged.
        """
        with self.client.messages.stream(**request_kwargs) as stream:
            return stream.get_final_message()

    def _build_model_response(self, response, resolved_model_name: str, thinking_mode: Optional[str]) -> ModelResponse:
        """Convert an Anthropic Message into a PAL ModelResponse."""
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        for block in getattr(response, "content", None) or []:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(getattr(block, "text", "") or "")
            elif block_type == "thinking":
                thinking_parts.append(getattr(block, "thinking", "") or "")

        usage: dict[str, int] = {}
        raw_usage = getattr(response, "usage", None)
        if raw_usage is not None:
            input_tokens = getattr(raw_usage, "input_tokens", None)
            output_tokens = getattr(raw_usage, "output_tokens", None)
            if input_tokens is not None:
                usage["input_tokens"] = input_tokens
            if output_tokens is not None:
                usage["output_tokens"] = output_tokens
            if input_tokens is not None and output_tokens is not None:
                usage["total_tokens"] = input_tokens + output_tokens

        return ModelResponse(
            content="".join(text_parts),
            usage=usage,
            model_name=resolved_model_name,
            friendly_name=self.FRIENDLY_NAME,
            provider=ProviderType.ANTHROPIC,
            metadata={
                "thinking_mode": thinking_mode,
                "finish_reason": getattr(response, "stop_reason", None),
                "has_thinking": bool(thinking_parts),
            },
        )

    def _process_image(self, image_path: str) -> Optional[dict]:
        """Build an Anthropic image content block from a path or data URL."""
        try:
            image_bytes, mime_type = validate_image(image_path)
        except ValueError as exc:
            logger.warning(str(exc))
            return None
        except Exception as exc:  # noqa: BLE001 - never fail the whole request on one image
            logger.error("Error processing image %s: %s", image_path, exc)
            return None

        if image_path.startswith("data:"):
            _, data = image_path.split(",", 1)
        else:
            data = base64.b64encode(image_bytes).decode()

        return {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": data}}

    def _is_error_retryable(self, error: Exception) -> bool:
        """Retry on transient Anthropic errors; never retry rate limits or client errors."""
        try:
            import anthropic
        except Exception:  # noqa: BLE001 - SDK should be installed, but degrade gracefully
            return super()._is_error_retryable(error)

        if isinstance(error, (anthropic.APITimeoutError, anthropic.APIConnectionError, anthropic.InternalServerError)):
            return True
        if isinstance(error, anthropic.RateLimitError):
            return False
        status = getattr(error, "status_code", None)
        if status in (500, 502, 503, 504, 529):
            return True
        return False


# Load registry data at import time for registry consumers.
AnthropicModelProvider._ensure_registry()
