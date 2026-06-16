# Direct Anthropic API Key Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a first-class Anthropic provider so PAL can talk directly to `api.anthropic.com` with an `ANTHROPIC_API_KEY`, instead of only reaching Claude models through OpenRouter or DIAL.

**Architecture:** A native provider (`AnthropicModelProvider`) subclasses `ModelProvider` directly and uses the official `anthropic` Python SDK against the native Messages API — *not* Anthropic's OpenAI-compatible endpoint. The OpenAI-compat endpoint is explicitly documented by Anthropic as a non-production compatibility layer that **ignores `reasoning_effort`, does not return thinking blocks, has no token counting, and drops prompt caching** — all of which PAL depends on. Model capabilities load from `conf/anthropic_models.json` via the existing `RegistryBackedProviderMixin` pattern (mirrors Gemini/X.AI). The provider is wired into the registry, server bootstrap, restriction service, and `listmodels`.

**Tech Stack:** Python 3.9+, `anthropic` SDK (Messages API), existing PAL provider/registry abstraction, pytest.

---

## Key design decisions (read before starting)

1. **Native SDK, not OpenAI-compat.** Decided from web research: the OpenAI-compatible shim cannot carry thinking config or token counts, which are core PAL features. We build on `anthropic.Anthropic(...).messages.create(...)`.

2. **Thinking has two schemes; branch by capability.** Newer Claude models (Opus 4.6/4.7/4.8, Sonnet 4.6, Fable 5) use **adaptive** thinking (`thinking={"type":"adaptive"}` + an `effort` string) and *reject* the legacy budget scheme with HTTP 400. Older thinking-capable models (Haiku 4.5, Sonnet 4.5, Opus 4.5) use the **budget** scheme (`thinking={"type":"enabled","budget_tokens":N}`). We distinguish them with the existing `default_reasoning_effort` field in `ModelCapabilities`: if it is set, the model is adaptive; otherwise budget. Adaptive params are sent via the SDK's `extra_body` escape hatch so we are not coupled to a specific SDK type-literal version.

3. **No bare `opus`/`sonnet`/`haiku` aliases (deliberate, deferred).** OpenRouter already claims the bare `opus`/`sonnet`/`haiku` and dotless forms like `opus4.5`; DIAL claims `opus-4.1`/`sonnet-4.1`-style aliases. The Anthropic manifest uses only version-specific (`opus-4.8`, `sonnet-4.6`, `haiku-4.5`) and `claude-`-prefixed aliases, which were **verified to not overlap** with any existing OpenRouter/DIAL alias. We intentionally do **not** hijack the bare single-word aliases in this plan; that is a follow-up UX decision. Anthropic sits *above* OpenRouter/DIAL in priority order, so if a shared alias is ever introduced, the direct Anthropic provider would win — the desired behavior.

4. **Anthropic is NOT registered in the global test fixture (conservative choice, not a collision workaround).** Because the Anthropic aliases were verified to not overlap with OpenRouter/DIAL (see #3), global registration would actually be safe today. But no Anthropic unit test needs registry-routed resolution — every test instantiates the provider directly (`AnthropicModelProvider("test-key")`), exactly like `tests/test_xai_provider.py` does — so we keep Anthropic out of the global fixture to minimize blast radius on the existing suite. We only add `ANTHROPIC_ALLOWED_MODELS` to the restriction-clearing fixture for isolation.

---

## File Structure

**Create:**
- `conf/anthropic_models.json` — capability manifest for direct Anthropic models.
- `providers/registries/anthropic.py` — `AnthropicModelRegistry` (JSON loader).
- `providers/anthropic.py` — `AnthropicModelProvider` (native SDK provider).
- `tests/test_anthropic_registry.py` — registry/manifest validation.
- `tests/test_anthropic_provider.py` — provider behavior + mocked-client request/response tests.

**Modify:**
- `providers/shared/provider_type.py` — add `ANTHROPIC` enum member.
- `providers/registries/__init__.py` — export `AnthropicModelRegistry`.
- `providers/__init__.py` — export `AnthropicModelProvider`.
- `providers/registry.py` — priority order + API-key map.
- `server.py` — `configure_providers()` detection + registration + error text + restriction validation list + debug key list + import.
- `utils/model_restrictions.py` — `ENV_VARS` map + module docstring.
- `tools/listmodels.py` — `provider_info` map.
- `tests/conftest.py` — clear `ANTHROPIC_ALLOWED_MODELS` between tests.
- `.env.example` — document `ANTHROPIC_API_KEY` + `ANTHROPIC_ALLOWED_MODELS`.
- `requirements.txt` — add `anthropic` SDK.

---

## Preconditions

- [ ] **Step 0: Activate venv and confirm a clean baseline.**

Run:
```bash
source venv/bin/activate
python -m pytest tests/ -q -m "not integration"
```
Expected: all tests pass (this is the pre-change baseline). If anything fails before you start, stop and report.

---

### Task 1: Add the `ANTHROPIC` provider type

**Files:**
- Modify: `providers/shared/provider_type.py:8-17`
- Test: `tests/test_provider_type_anthropic.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_provider_type_anthropic.py`:
```python
"""Tests for the ANTHROPIC provider type member."""

from providers.shared import ProviderType


def test_anthropic_provider_type_exists():
    assert ProviderType.ANTHROPIC.value == "anthropic"


def test_anthropic_provider_type_is_unique():
    values = [member.value for member in ProviderType]
    assert values.count("anthropic") == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_provider_type_anthropic.py -v`
Expected: FAIL with `AttributeError: ANTHROPIC` (member does not exist yet).

- [ ] **Step 3: Add the enum member**

In `providers/shared/provider_type.py`, add `ANTHROPIC` to the enum body (after `GOOGLE`/`OPENAI` group, keep it alphabetical-ish near the top):
```python
class ProviderType(Enum):
    """Canonical identifiers for every supported provider backend."""

    GOOGLE = "google"
    OPENAI = "openai"
    AZURE = "azure"
    ANTHROPIC = "anthropic"
    XAI = "xai"
    OPENROUTER = "openrouter"
    CUSTOM = "custom"
    DIAL = "dial"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_provider_type_anthropic.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add providers/shared/provider_type.py tests/test_provider_type_anthropic.py
git commit -m "feat: add ANTHROPIC provider type"
```

---

### Task 2: Add the `anthropic` SDK dependency

**Files:**
- Modify: `requirements.txt:1-6`

- [ ] **Step 1: Add the dependency**

In `requirements.txt`, add the `anthropic` line under the other SDK deps:
```text
mcp>=1.0.0
google-genai>=1.19.0
openai>=1.55.2  # Minimum version for httpx 0.28.0 compatibility
anthropic>=0.40.0  # Direct Anthropic Messages API support
pydantic>=2.0.0
python-dotenv>=1.0.0
importlib-resources>=5.0.0; python_version<"3.9"
```

- [ ] **Step 2: Install it**

Run: `pip install -r requirements.txt`
Expected: `anthropic` installs successfully.

- [ ] **Step 3: Verify importability and capture the installed version**

Run: `python -c "import anthropic; print(anthropic.__version__)"`
Expected: prints a version string (e.g. `0.40.0` or newer). Note the version — Step is informational; the provider uses only stable surface (`messages.create`, `messages.count_tokens`, exception classes, `extra_body`).

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "build: add anthropic SDK dependency"
```

---

### Task 3: Create the Anthropic model manifest and registry

**Files:**
- Create: `conf/anthropic_models.json`
- Create: `providers/registries/anthropic.py`
- Modify: `providers/registries/__init__.py:1-19`
- Test: `tests/test_anthropic_registry.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_anthropic_registry.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_anthropic_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'providers.registries.anthropic'`.

- [ ] **Step 3: Create the manifest `conf/anthropic_models.json`**

```json
{
  "_README": {
    "description": "Model metadata for direct Anthropic (Claude) API access via the official anthropic SDK.",
    "documentation": "https://github.com/BeehiveInnovations/pal-mcp-server/blob/main/docs/custom_models.md",
    "usage": "Models listed here are exposed directly through the Anthropic provider (ANTHROPIC_API_KEY). Aliases are case-insensitive.",
    "field_notes": "Matches providers/shared/model_capabilities.py.",
    "field_descriptions": {
      "model_name": "The Anthropic API model id (e.g., 'claude-opus-4-8')",
      "aliases": "Array of short names users can type instead of the full model name",
      "context_window": "Total number of tokens the model can process (input + output combined)",
      "max_output_tokens": "Maximum number of tokens the model can generate in a single response",
      "max_thinking_tokens": "Budget-scheme thinking ceiling (0 for adaptive-thinking models, which use default_reasoning_effort instead)",
      "supports_extended_thinking": "Whether the model supports extended reasoning",
      "default_reasoning_effort": "Set to an effort string (e.g. 'high') for ADAPTIVE-thinking models; omit for budget-scheme models",
      "supports_json_mode": "Whether the model can guarantee valid JSON output",
      "supports_function_calling": "Whether the model supports tool/function calling",
      "supports_images": "Whether the model can process images",
      "max_image_size_mb": "Maximum total size in MB for all images combined",
      "supports_temperature": "Whether the model accepts a temperature parameter",
      "temperature_constraint": "'range', 'fixed', 'discrete', or omit for default range",
      "intelligence_score": "1-20 human rating used as the primary signal for auto-mode model ordering",
      "description": "Human-readable description of the model"
    }
  },
  "models": [
    {
      "model_name": "claude-fable-5",
      "friendly_name": "Anthropic (Claude Fable 5)",
      "aliases": ["fable", "fable-5", "claude-fable-5"],
      "intelligence_score": 19,
      "description": "Claude Fable 5 (1M context) - flagship multimodal model with always-on adaptive thinking",
      "context_window": 1000000,
      "max_output_tokens": 128000,
      "max_thinking_tokens": 0,
      "supports_extended_thinking": true,
      "default_reasoning_effort": "high",
      "supports_system_prompts": true,
      "supports_streaming": true,
      "supports_function_calling": true,
      "supports_json_mode": true,
      "supports_images": true,
      "supports_temperature": true,
      "max_image_size_mb": 5.0
    },
    {
      "model_name": "claude-opus-4-8",
      "friendly_name": "Anthropic (Claude Opus 4.8)",
      "aliases": ["opus-4.8", "claude-opus-4.8", "claude-opus-4-8"],
      "intelligence_score": 18,
      "description": "Claude Opus 4.8 (1M context) - highest-capability Claude with adaptive thinking",
      "context_window": 1000000,
      "max_output_tokens": 128000,
      "max_thinking_tokens": 0,
      "supports_extended_thinking": true,
      "default_reasoning_effort": "high",
      "supports_system_prompts": true,
      "supports_streaming": true,
      "supports_function_calling": true,
      "supports_json_mode": true,
      "supports_images": true,
      "supports_temperature": true,
      "max_image_size_mb": 5.0
    },
    {
      "model_name": "claude-opus-4-7",
      "friendly_name": "Anthropic (Claude Opus 4.7)",
      "aliases": ["opus-4.7", "claude-opus-4.7"],
      "intelligence_score": 17,
      "description": "Claude Opus 4.7 (1M context) - adaptive-thinking flagship",
      "context_window": 1000000,
      "max_output_tokens": 128000,
      "max_thinking_tokens": 0,
      "supports_extended_thinking": true,
      "default_reasoning_effort": "high",
      "supports_system_prompts": true,
      "supports_streaming": true,
      "supports_function_calling": true,
      "supports_json_mode": true,
      "supports_images": true,
      "supports_temperature": true,
      "max_image_size_mb": 5.0
    },
    {
      "model_name": "claude-opus-4-6",
      "friendly_name": "Anthropic (Claude Opus 4.6)",
      "aliases": ["opus-4.6", "claude-opus-4.6"],
      "intelligence_score": 16,
      "description": "Claude Opus 4.6 (1M context) - adaptive thinking",
      "context_window": 1000000,
      "max_output_tokens": 128000,
      "max_thinking_tokens": 0,
      "supports_extended_thinking": true,
      "default_reasoning_effort": "high",
      "supports_system_prompts": true,
      "supports_streaming": true,
      "supports_function_calling": true,
      "supports_json_mode": true,
      "supports_images": true,
      "supports_temperature": true,
      "max_image_size_mb": 5.0
    },
    {
      "model_name": "claude-sonnet-4-6",
      "friendly_name": "Anthropic (Claude Sonnet 4.6)",
      "aliases": ["sonnet-4.6", "claude-sonnet-4.6"],
      "intelligence_score": 15,
      "description": "Claude Sonnet 4.6 (1M context) - balanced model with adaptive thinking",
      "context_window": 1000000,
      "max_output_tokens": 64000,
      "max_thinking_tokens": 0,
      "supports_extended_thinking": true,
      "default_reasoning_effort": "high",
      "supports_system_prompts": true,
      "supports_streaming": true,
      "supports_function_calling": true,
      "supports_json_mode": true,
      "supports_images": true,
      "supports_temperature": true,
      "max_image_size_mb": 5.0
    },
    {
      "model_name": "claude-opus-4-5-20251101",
      "friendly_name": "Anthropic (Claude Opus 4.5)",
      "aliases": ["opus-4.5", "claude-opus-4.5"],
      "intelligence_score": 15,
      "description": "Claude Opus 4.5 (200K context) - budget-scheme extended thinking",
      "context_window": 200000,
      "max_output_tokens": 64000,
      "max_thinking_tokens": 32000,
      "supports_extended_thinking": true,
      "supports_system_prompts": true,
      "supports_streaming": true,
      "supports_function_calling": true,
      "supports_json_mode": true,
      "supports_images": true,
      "supports_temperature": true,
      "max_image_size_mb": 5.0
    },
    {
      "model_name": "claude-sonnet-4-5-20250929",
      "friendly_name": "Anthropic (Claude Sonnet 4.5)",
      "aliases": ["sonnet-4.5", "claude-sonnet-4.5"],
      "intelligence_score": 13,
      "description": "Claude Sonnet 4.5 (200K context) - budget-scheme extended thinking",
      "context_window": 200000,
      "max_output_tokens": 64000,
      "max_thinking_tokens": 32000,
      "supports_extended_thinking": true,
      "supports_system_prompts": true,
      "supports_streaming": true,
      "supports_function_calling": true,
      "supports_json_mode": true,
      "supports_images": true,
      "supports_temperature": true,
      "max_image_size_mb": 5.0
    },
    {
      "model_name": "claude-haiku-4-5-20251001",
      "friendly_name": "Anthropic (Claude Haiku 4.5)",
      "aliases": ["haiku-4.5", "claude-haiku-4.5", "claude-haiku-4-5"],
      "intelligence_score": 11,
      "description": "Claude Haiku 4.5 (200K context) - fast, low-cost; budget-scheme thinking",
      "context_window": 200000,
      "max_output_tokens": 64000,
      "max_thinking_tokens": 24000,
      "supports_extended_thinking": true,
      "supports_system_prompts": true,
      "supports_streaming": true,
      "supports_function_calling": true,
      "supports_json_mode": true,
      "supports_images": true,
      "supports_temperature": true,
      "max_image_size_mb": 5.0
    }
  ]
}
```

- [ ] **Step 4: Create `providers/registries/anthropic.py`**

```python
"""Registry loader for Anthropic model capabilities."""

from __future__ import annotations

from ..shared import ProviderType
from .base import CapabilityModelRegistry


class AnthropicModelRegistry(CapabilityModelRegistry):
    """Capability registry backed by ``conf/anthropic_models.json``."""

    def __init__(self, config_path: str | None = None) -> None:
        super().__init__(
            env_var_name="ANTHROPIC_MODELS_CONFIG_PATH",
            default_filename="anthropic_models.json",
            provider=ProviderType.ANTHROPIC,
            friendly_prefix="Anthropic ({model})",
            config_path=config_path,
        )
```

- [ ] **Step 5: Export the registry**

In `providers/registries/__init__.py`, add the import and `__all__` entry (keep alphabetical):
```python
"""Registry implementations for provider capability manifests."""

from .anthropic import AnthropicModelRegistry
from .azure import AzureModelRegistry
from .custom import CustomEndpointModelRegistry
from .dial import DialModelRegistry
from .gemini import GeminiModelRegistry
from .openai import OpenAIModelRegistry
from .openrouter import OpenRouterModelRegistry
from .xai import XAIModelRegistry

__all__ = [
    "AnthropicModelRegistry",
    "AzureModelRegistry",
    "CustomEndpointModelRegistry",
    "DialModelRegistry",
    "GeminiModelRegistry",
    "OpenAIModelRegistry",
    "OpenRouterModelRegistry",
    "XAIModelRegistry",
]
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_anthropic_registry.py -v`
Expected: PASS (all 5 tests).

- [ ] **Step 7: Commit**

```bash
git add conf/anthropic_models.json providers/registries/anthropic.py providers/registries/__init__.py tests/test_anthropic_registry.py
git commit -m "feat: add Anthropic capability registry and model manifest"
```

---

### Task 4: Create the Anthropic provider scaffold (identity + capabilities)

This task creates the class with everything *except* a working `generate_content` (a `NotImplementedError` stub satisfies the `ABC` so the class is instantiable and the identity/capability tests can run). `generate_content` is implemented in Task 5.

**Files:**
- Create: `providers/anthropic.py`
- Modify: `providers/__init__.py:1-22`
- Test: `tests/test_anthropic_provider.py` (create — identity/capability/restriction tests)

- [ ] **Step 1: Write the failing test**

Create `tests/test_anthropic_provider.py`:
```python
"""Tests for the direct Anthropic provider."""

import os
from unittest.mock import MagicMock, patch

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
```

> Note: the provider-level `ANTHROPIC_ALLOWED_MODELS` restriction test lives in Task 8, **not** here. Restriction enforcement requires `ProviderType.ANTHROPIC` to be present in `ModelRestrictionService.ENV_VARS`, which is added in Task 8. If that test ran now, `is_allowed` would return `True` for every model (the provider is absent from the restriction map), so the "blocked" assertions would fail. The `os`/`patch` imports above are still used by the generate-content tests in Task 5.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_anthropic_provider.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'providers.anthropic'`.

- [ ] **Step 3: Create `providers/anthropic.py` (scaffold with stubbed generate_content)**

```python
"""Anthropic (Claude) model provider implementation.

Talks directly to the Anthropic Messages API via the official ``anthropic``
Python SDK rather than routing Claude through OpenRouter or DIAL. This gives
PAL first-class access to native features the OpenAI-compatible shim does not
expose: extended/adaptive thinking, accurate token counting, and prompt caching.
"""

import base64
import logging
from typing import TYPE_CHECKING, ClassVar, Optional

if TYPE_CHECKING:
    from tools.models import ToolModelCategory

from utils.image_utils import validate_image

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
```

- [ ] **Step 4: Export the provider**

In `providers/__init__.py`, add the import and `__all__` entry:
```python
"""Model provider abstractions for supporting multiple AI providers."""

from .anthropic import AnthropicModelProvider
from .azure_openai import AzureOpenAIProvider
from .base import ModelProvider
from .gemini import GeminiModelProvider
from .openai import OpenAIModelProvider
from .openai_compatible import OpenAICompatibleProvider
from .openrouter import OpenRouterProvider
from .registry import ModelProviderRegistry
from .shared import ModelCapabilities, ModelResponse

__all__ = [
    "ModelProvider",
    "ModelResponse",
    "ModelCapabilities",
    "ModelProviderRegistry",
    "AnthropicModelProvider",
    "AzureOpenAIProvider",
    "GeminiModelProvider",
    "OpenAIModelProvider",
    "OpenAICompatibleProvider",
    "OpenRouterProvider",
]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_anthropic_provider.py -v`
Expected: PASS (identity/capability/validation tests). The `generate_content` stub is not exercised by these tests, and the restriction test is deferred to Task 8.

- [ ] **Step 6: Commit**

```bash
git add providers/anthropic.py providers/__init__.py tests/test_anthropic_provider.py
git commit -m "feat: add Anthropic provider scaffold (identity + capabilities)"
```

---

### Task 5: Implement `generate_content`, thinking, token counting, and helpers

**Files:**
- Modify: `providers/anthropic.py` (replace the `generate_content` stub; add helpers)
- Test: `tests/test_anthropic_provider.py` (append mocked-client tests)

- [ ] **Step 1: Write the failing tests (append to `tests/test_anthropic_provider.py`)**

```python
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

        provider.generate_content(
            prompt="think", model_name="claude-haiku-4-5-20251001", thinking_mode="medium"
        )
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
        b1 = MagicMock(); b1.type = "text"; b1.text = "foo "
        thinking_block = MagicMock(); thinking_block.type = "thinking"; thinking_block.thinking = "reasoning"
        b2 = MagicMock(); b2.type = "text"; b2.text = "bar"
        msg = MagicMock()
        msg.content = [thinking_block, b1, b2]
        msg.stop_reason = "end_turn"
        msg.usage = MagicMock(); msg.usage.input_tokens = 1; msg.usage.output_tokens = 2
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_anthropic_provider.py::TestAnthropicGenerateContent -v`
Expected: FAIL with `NotImplementedError: generate_content is implemented in Task 5`.

- [ ] **Step 3: Replace the `generate_content` stub and add helpers**

In `providers/anthropic.py`, replace the `generate_content` stub with the following implementation and add the helper methods below it (before the module-level `_ensure_registry()` call):

```python
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

        if thinking_params:
            request_kwargs.update(thinking_params)
            # Extended thinking requires temperature == 1.0 on Anthropic models.
            request_kwargs["temperature"] = 1.0
        elif capabilities.supports_temperature:
            # Anthropic accepts temperature in [0, 1]; clamp to be safe.
            request_kwargs["temperature"] = min(max(temperature, 0.0), 1.0)

        max_retries = 4
        retry_delays = [1, 3, 5, 8]
        attempt_counter = {"value": 0}
        active_thinking_mode = thinking_mode if thinking_params else None

        def _attempt() -> ModelResponse:
            attempt_counter["value"] += 1
            response = self.client.messages.create(**request_kwargs)
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

    def _build_model_response(
        self, response, resolved_model_name: str, thinking_mode: Optional[str]
    ) -> ModelResponse:
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
```

> **Implementation note on adaptive thinking:** the adaptive scheme (`thinking={"type":"adaptive"}` + `output_config.effort`) is sent through the SDK's `extra_body` rather than typed kwargs, so the code does not depend on a specific SDK type-literal version. The unit tests pin the exact body shape. The live-API behavior of these params should be confirmed during integration testing (Task 11), not in unit tests.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_anthropic_provider.py -v`
Expected: PASS (both identity and generate-content test classes).

- [ ] **Step 5: Commit**

```bash
git add providers/anthropic.py tests/test_anthropic_provider.py
git commit -m "feat: implement Anthropic generate_content, thinking, and token counting"
```

---

### Task 6: Wire the provider into the registry (priority + API key map)

**Files:**
- Modify: `providers/registry.py:38-46` (priority order)
- Modify: `providers/registry.py:334-342` (API key map)
- Test: `tests/test_anthropic_registry_wiring.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_anthropic_registry_wiring.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_anthropic_registry_wiring.py -v`
Expected: FAIL — `ANTHROPIC not in PROVIDER_PRIORITY_ORDER` (first test) and key lookup returns `None` (second).

- [ ] **Step 3: Add Anthropic to the priority order**

In `providers/registry.py`, update `PROVIDER_PRIORITY_ORDER` (lines 38-46) to insert `ANTHROPIC` after `XAI` and before `DIAL`:
```python
    PROVIDER_PRIORITY_ORDER = [
        ProviderType.GOOGLE,  # Direct Gemini access
        ProviderType.OPENAI,  # Direct OpenAI access
        ProviderType.AZURE,  # Azure-hosted OpenAI deployments
        ProviderType.XAI,  # Direct X.AI GROK access
        ProviderType.ANTHROPIC,  # Direct Anthropic (Claude) access
        ProviderType.DIAL,  # DIAL unified API access
        ProviderType.CUSTOM,  # Local/self-hosted models
        ProviderType.OPENROUTER,  # Catch-all for cloud models
    ]
```

- [ ] **Step 4: Add Anthropic to the API key map**

In `providers/registry.py`, update `key_mapping` in `_get_api_key_for_provider` (lines 334-342):
```python
        key_mapping = {
            ProviderType.GOOGLE: "GEMINI_API_KEY",
            ProviderType.OPENAI: "OPENAI_API_KEY",
            ProviderType.AZURE: "AZURE_OPENAI_API_KEY",
            ProviderType.XAI: "XAI_API_KEY",
            ProviderType.ANTHROPIC: "ANTHROPIC_API_KEY",
            ProviderType.OPENROUTER: "OPENROUTER_API_KEY",
            ProviderType.CUSTOM: "CUSTOM_API_KEY",  # Can be empty for providers that don't need auth
            ProviderType.DIAL: "DIAL_API_KEY",
        }
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_anthropic_registry_wiring.py -v`
Expected: PASS. The generic `else` branch in `get_provider` (lines 142-146) constructs the provider with just `api_key`, so no special case is needed.

- [ ] **Step 6: Commit**

```bash
git add providers/registry.py tests/test_anthropic_registry_wiring.py
git commit -m "feat: register Anthropic in provider priority order and key map"
```

---

### Task 7: Wire the provider into server bootstrap

**Files:**
- Modify: `server.py:390` (debug key list)
- Modify: `server.py:394-402` (imports)
- Modify: `server.py:451-463` (detection block — add after X.AI / before DIAL)
- Modify: `server.py:512-519` (registration block)
- Modify: `server.py:544-553` (no-providers error message)
- Modify: `server.py:603` (restriction validation list)

- [ ] **Step 1: Add the debug key to `api_keys_to_check` (line 390)**

```python
    api_keys_to_check = [
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "GEMINI_API_KEY",
        "XAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "CUSTOM_API_URL",
    ]
```

- [ ] **Step 2: Add the provider import (after the X.AI import at line 402)**

In the import block inside `configure_providers()`, add:
```python
    from providers.anthropic import AnthropicModelProvider
```
Place it next to `from providers.xai import XAIModelProvider`.

- [ ] **Step 3: Add the detection block (after the X.AI detection, before DIAL detection ~line 457)**

```python
    # Check for Anthropic API key
    anthropic_key = get_env("ANTHROPIC_API_KEY")
    if anthropic_key and anthropic_key != "your_anthropic_api_key_here":
        valid_providers.append("Anthropic")
        has_native_apis = True
        logger.info("Anthropic API key found - Claude models available")
```

- [ ] **Step 4: Add the registration block (inside `if has_native_apis:`, after the X.AI registration ~line 515)**

```python
        if anthropic_key and anthropic_key != "your_anthropic_api_key_here":
            ModelProviderRegistry.register_provider(ProviderType.ANTHROPIC, AnthropicModelProvider)
            registered_providers.append(ProviderType.ANTHROPIC.value)
            logger.debug(f"Registered provider: {ProviderType.ANTHROPIC.value}")
```

- [ ] **Step 5: Add Anthropic to the no-providers error message (lines 544-553)**

```python
    if not valid_providers:
        raise ValueError(
            "At least one API configuration is required. Please set either:\n"
            "- GEMINI_API_KEY for Gemini models\n"
            "- OPENAI_API_KEY for OpenAI models\n"
            "- XAI_API_KEY for X.AI GROK models\n"
            "- ANTHROPIC_API_KEY for Anthropic Claude models\n"
            "- DIAL_API_KEY for DIAL models\n"
            "- OPENROUTER_API_KEY for OpenRouter (multiple models)\n"
            "- CUSTOM_API_URL for local models (Ollama, vLLM, etc.)"
        )
```

- [ ] **Step 6: Add Anthropic to the restriction validation list (line 603)**

```python
        provider_types_to_validate = [
            ProviderType.GOOGLE,
            ProviderType.OPENAI,
            ProviderType.XAI,
            ProviderType.ANTHROPIC,
            ProviderType.DIAL,
        ]
```

- [ ] **Step 7: Verify server import still works**

Run: `python -c "import server; print('ok')"`
Expected: prints `ok` with no import errors.

- [ ] **Step 8: Run the server-related tests**

Run: `python -m pytest tests/ -q -m "not integration" -k "server or provider or config"`
Expected: PASS (no regressions).

- [ ] **Step 9: Commit**

```bash
git add server.py
git commit -m "feat: wire Anthropic provider into server bootstrap"
```

---

### Task 8: Add restriction support and test isolation

**Files:**
- Modify: `utils/model_restrictions.py:9-21` (docstring) and `:51-57` (`ENV_VARS`)
- Modify: `tests/conftest.py:182-191` (`clear_model_restriction_env`)
- Test: `tests/test_anthropic_restrictions.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_anthropic_restrictions.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_anthropic_restrictions.py -v`
Expected: FAIL — `test_anthropic_env_var_registered` raises `KeyError: ProviderType.ANTHROPIC`, and the other two fail on the "blocked" assertions because `ANTHROPIC_ALLOWED_MODELS` is not yet loaded (so every model is allowed).

- [ ] **Step 3: Add Anthropic to `ENV_VARS` (lines 51-57)**

```python
    ENV_VARS = {
        ProviderType.OPENAI: "OPENAI_ALLOWED_MODELS",
        ProviderType.GOOGLE: "GOOGLE_ALLOWED_MODELS",
        ProviderType.XAI: "XAI_ALLOWED_MODELS",
        ProviderType.ANTHROPIC: "ANTHROPIC_ALLOWED_MODELS",
        ProviderType.OPENROUTER: "OPENROUTER_ALLOWED_MODELS",
        ProviderType.DIAL: "DIAL_ALLOWED_MODELS",
    }
```

- [ ] **Step 4: Update the module docstring (lines 9-21)**

Add the `ANTHROPIC_ALLOWED_MODELS` line to the Environment Variables list and the example:
```python
Environment Variables:
- OPENAI_ALLOWED_MODELS: Comma-separated list of allowed OpenAI models
- GOOGLE_ALLOWED_MODELS: Comma-separated list of allowed Gemini models
- XAI_ALLOWED_MODELS: Comma-separated list of allowed X.AI GROK models
- ANTHROPIC_ALLOWED_MODELS: Comma-separated list of allowed Anthropic Claude models
- OPENROUTER_ALLOWED_MODELS: Comma-separated list of allowed OpenRouter models
- DIAL_ALLOWED_MODELS: Comma-separated list of allowed DIAL models

Example:
    OPENAI_ALLOWED_MODELS=o3-mini,o4-mini
    GOOGLE_ALLOWED_MODELS=flash
    XAI_ALLOWED_MODELS=grok-4,grok-4.1-fast-reasoning
    ANTHROPIC_ALLOWED_MODELS=claude-opus-4-8,claude-sonnet-4-6
    OPENROUTER_ALLOWED_MODELS=opus,sonnet,mistral
```

- [ ] **Step 5: Add `ANTHROPIC_ALLOWED_MODELS` to the test-isolation fixture (`tests/conftest.py` lines 182-191)**

```python
    restriction_vars = [
        "OPENAI_ALLOWED_MODELS",
        "GOOGLE_ALLOWED_MODELS",
        "XAI_ALLOWED_MODELS",
        "ANTHROPIC_ALLOWED_MODELS",
        "OPENROUTER_ALLOWED_MODELS",
        "DIAL_ALLOWED_MODELS",
    ]
```

> Note: we deliberately do **not** add Anthropic to the global provider-registration block or to `_set_dummy_keys_if_missing` in conftest (see Key design decision #4). This is a conservative blast-radius choice — Anthropic's aliases were verified to not overlap with OpenRouter/DIAL, and all Anthropic unit tests instantiate the provider directly, so global registration is unnecessary.

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_anthropic_restrictions.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add utils/model_restrictions.py tests/conftest.py tests/test_anthropic_restrictions.py
git commit -m "feat: add ANTHROPIC_ALLOWED_MODELS restriction support"
```

---

### Task 9: Surface Anthropic in `listmodels`

**Files:**
- Modify: `tools/listmodels.py:100-106` (`provider_info` map)
- Test: verify existing `tests/test_listmodels.py` still passes; add an assertion.

- [ ] **Step 1: Add Anthropic to `provider_info` (lines 100-106)**

```python
        provider_info = {
            ProviderType.GOOGLE: {"name": "Google Gemini", "env_key": "GEMINI_API_KEY"},
            ProviderType.OPENAI: {"name": "OpenAI", "env_key": "OPENAI_API_KEY"},
            ProviderType.AZURE: {"name": "Azure OpenAI", "env_key": "AZURE_OPENAI_API_KEY"},
            ProviderType.XAI: {"name": "X.AI (Grok)", "env_key": "XAI_API_KEY"},
            ProviderType.ANTHROPIC: {"name": "Anthropic Claude", "env_key": "ANTHROPIC_API_KEY"},
            ProviderType.DIAL: {"name": "AI DIAL", "env_key": "DIAL_API_KEY"},
        }
```

- [ ] **Step 2: Run the listmodels tests to check for regressions**

Run: `python -m pytest tests/test_listmodels.py -v`
Expected: PASS. If a test asserts an exact provider count or full output snapshot and now fails, update that expectation to include the new "Anthropic Claude" section (it renders as `## Anthropic Claude ❌` / `**Status**: Not configured (set ANTHROPIC_API_KEY)` when no key is present). Re-run until green.

- [ ] **Step 3: Add a focused assertion (append to `tests/test_listmodels.py`)**

Find the async execution pattern already used in that file (it calls the tool's `execute` and inspects the returned text) and add a test mirroring it. If the file uses a helper like `_run_listmodels()` / direct `asyncio.run`, reuse it. Example shape (adapt to the file's existing imports and helpers):
```python
import asyncio

from tools.listmodels import ListModelsTool


def test_listmodels_includes_anthropic_section():
    tool = ListModelsTool()
    result = asyncio.run(tool.execute({}))
    text = result[0].text
    assert "Anthropic Claude" in text
    assert "ANTHROPIC_API_KEY" in text
```

- [ ] **Step 4: Run the new test**

Run: `python -m pytest tests/test_listmodels.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/listmodels.py tests/test_listmodels.py
git commit -m "feat: surface Anthropic provider in listmodels"
```

---

### Task 10: Document `ANTHROPIC_API_KEY` in `.env.example`

**Files:**
- Modify: `.env.example` (after the X.AI key block ~line 30, before the DIAL block; and the restriction examples ~lines 132-145)

- [ ] **Step 1: Add the API key block (after line 30)**

```text
# Get your X.AI API key from: https://console.x.ai/
XAI_API_KEY=your_xai_api_key_here

# Get your Anthropic API key from: https://console.anthropic.com/
# Direct Claude access (Messages API). Takes priority over OpenRouter/DIAL for shared Claude aliases.
ANTHROPIC_API_KEY=your_anthropic_api_key_here
# ANTHROPIC_MODELS_CONFIG_PATH=                # Optional: path to a custom anthropic_models.json
```

- [ ] **Step 2: Add the restriction examples (in the `*_ALLOWED_MODELS` comment block ~lines 132-145)**

Add an example line near the other provider examples:
```text
#   ANTHROPIC_ALLOWED_MODELS=claude-opus-4-8,claude-sonnet-4-6   # Only allow specific Claude models
```
And add the commented blank-value line alongside the others:
```text
# ANTHROPIC_ALLOWED_MODELS=
```

- [ ] **Step 3: Commit**

```bash
git add .env.example
git commit -m "docs: document ANTHROPIC_API_KEY and restrictions in .env.example"
```

---

### Task 11: Full verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full unit suite**

Run: `python -m pytest tests/ -q -m "not integration"`
Expected: all tests pass, including the new Anthropic tests and all pre-existing tests (no regressions in OpenRouter/DIAL alias-resolution tests).

- [ ] **Step 2: Run code quality checks**

Run: `./code_quality_checks.sh`
Expected: ruff, black, isort, and the unit suite all pass 100%.

- [ ] **Step 3: Smoke-test provider construction and listmodels with a key set**

Run:
```bash
ANTHROPIC_API_KEY=sk-ant-dummy python -c "
import server
server.configure_providers()
from providers.registry import ModelProviderRegistry
from providers.shared import ProviderType
p = ModelProviderRegistry.get_provider(ProviderType.ANTHROPIC)
print('provider:', p.get_provider_type().value)
print('resolves opus-4.8 ->', p._resolve_model_name('opus-4.8'))
print('models:', p.list_models()[:5])
"
```
Expected: prints `provider: anthropic`, `resolves opus-4.8 -> claude-opus-4-8`, and a list of model names. (No real network call is made — the client is lazy and `generate_content` is not invoked.)

- [ ] **Step 4 (optional, requires a real key): live integration check**

If a real `ANTHROPIC_API_KEY` is available, verify the live Messages API contract — especially that the adaptive-thinking `extra_body` shape and the budget-thinking param are accepted by the current API. Run:
```bash
ANTHROPIC_API_KEY=<real-key> python -c "
from providers.anthropic import AnthropicModelProvider
p = AnthropicModelProvider('<real-key>')
r = p.generate_content(prompt='Reply with the single word: pong', model_name='haiku-4.5', thinking_mode='off', max_output_tokens=16)
print('content:', repr(r.content)); print('usage:', r.usage)
r2 = p.generate_content(prompt='Think briefly, then say pong', model_name='opus-4.8', thinking_mode='high', max_output_tokens=2048)
print('adaptive content:', repr(r2.content)); print('has_thinking:', r2.metadata['has_thinking'])
"
```
Expected: both calls return non-empty content with usage populated. If the adaptive `extra_body` shape is rejected by the live API, adjust `_resolve_thinking_params` to match the installed SDK's documented adaptive-thinking parameters and re-run; the unit tests in Task 5 will need their asserted body shape updated to match.

- [ ] **Step 5: Final commit (if any quality-check auto-fixes were applied)**

```bash
git add -A
git commit -m "chore: apply code quality fixes for Anthropic provider"
```

---

## Self-Review (completed by plan author)

**Spec coverage** — every item from the unified research findings maps to a task:
- New `ProviderType` → Task 1. SDK dependency → Task 2. Manifest + registry → Task 3. Provider class + identity/capabilities → Task 4. `generate_content`/thinking/token-counting → Task 5. Registry priority + key map → Task 6. Server bootstrap → Task 7. Restrictions + test isolation → Task 8. `listmodels` → Task 9. `.env.example` → Task 10. Verification → Task 11.
- Native SDK vs OpenAI-compat decision → captured in Architecture + Key decision #1.
- Two-scheme thinking → Key decision #2 + `_resolve_thinking_params` (Task 5) + tests.
- Alias-collision/priority concern → Key decisions #3 and #4 + conftest handling (Task 8).

**Placeholder scan** — no `TBD`/`implement later`. All code blocks are complete; commands have expected output.

**Type/name consistency** — canonical ids used consistently (`claude-opus-4-8`, `claude-sonnet-4-6`, `claude-haiku-4-5-20251001`); `default_reasoning_effort` (existing `ModelCapabilities` field) is the adaptive/budget discriminator everywhere; `AnthropicModelProvider`/`AnthropicModelRegistry`/`ProviderType.ANTHROPIC`/`ANTHROPIC_API_KEY`/`ANTHROPIC_ALLOWED_MODELS`/`ANTHROPIC_MODELS_CONFIG_PATH` names match across manifest, provider, registry, server, restrictions, and tests. `ModelResponse`/`ModelCapabilities` fields used match `providers/shared/`.

**Known residual risk (flagged, not a blocker):** the exact wire shape of adaptive-thinking params (`thinking={"type":"adaptive"}` + `output_config.effort`) is the one piece sourced from web research rather than the local code. It is isolated in `_resolve_thinking_params`, sent via `extra_body` to avoid SDK-version coupling, pinned by unit tests, and explicitly re-verified against the live API in Task 11 Step 4.
