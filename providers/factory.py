"""Factory model provider using the official Python Droid SDK."""

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, ClassVar, Optional

from utils.env import get_env, get_env_bool
from utils.image_utils import validate_image

if TYPE_CHECKING:
    from tools.models import ToolModelCategory

from .base import ModelProvider
from .registries.factory import FactoryModelRegistry
from .registry_provider_mixin import RegistryBackedProviderMixin
from .shared import ModelCapabilities, ModelResponse, ProviderType

logger = logging.getLogger(__name__)


class FactoryModelProvider(RegistryBackedProviderMixin, ModelProvider):
    """Run Factory-hosted models through a local Droid SDK subprocess."""

    FRIENDLY_NAME = "Factory (Droid SDK)"
    REGISTRY_CLASS = FactoryModelRegistry
    MODEL_CAPABILITIES: ClassVar[dict[str, ModelCapabilities]] = {}

    PRIMARY_MODEL = "deepseek-v4-pro"
    REQUEST_TIMEOUT_SECONDS = 600.0

    THINKING_EFFORT = {
        "off": "OFF",
        "minimal": "LOW",
        "low": "LOW",
        "medium": "HIGH",
        "high": "HIGH",
        "max": "MAX",
    }

    DROID_PROCESS_ENVIRONMENT = frozenset(
        {
            "ALL_PROXY",
            "COLORTERM",
            "CURL_CA_BUNDLE",
            "GIT_SSL_CAINFO",
            "HOME",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "LANG",
            "LANGUAGE",
            "LC_ALL",
            "LC_CTYPE",
            "LOGNAME",
            "NIX_SSL_CERT_FILE",
            "NODE_EXTRA_CA_CERTS",
            "NO_COLOR",
            "NO_PROXY",
            "PATH",
            "REQUESTS_CA_BUNDLE",
            "SHELL",
            "SSL_CERT_DIR",
            "SSL_CERT_FILE",
            "TEMP",
            "TERM",
            "TMP",
            "TMPDIR",
            "TZ",
            "USER",
            "XDG_CACHE_HOME",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
            "XDG_STATE_HOME",
            "all_proxy",
            "http_proxy",
            "https_proxy",
            "no_proxy",
        }
    )

    def __init__(self, api_key: str = "", **kwargs):
        self._ensure_registry()
        super().__init__(api_key, **kwargs)
        self._invalidate_capability_cache()

    def get_provider_type(self) -> ProviderType:
        return ProviderType.FACTORY

    def generate_content(
        self,
        prompt: str,
        model_name: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_output_tokens: Optional[int] = None,
        thinking_mode: Optional[str] = "medium",
        images: Optional[list[str]] = None,
        **kwargs,
    ) -> ModelResponse:
        """Generate one model-only turn through Droid SDK."""
        self.validate_parameters(model_name, temperature)
        capabilities = self.get_capabilities(model_name)
        resolved_model = self._resolve_model_name(model_name)

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                result = executor.submit(
                    lambda: asyncio.run(
                        self._run_droid(
                            prompt=prompt,
                            model_name=resolved_model,
                            system_prompt=system_prompt,
                            thinking_mode=thinking_mode,
                            images=images if capabilities.supports_images else None,
                            max_image_size_mb=capabilities.max_image_size_mb,
                        )
                    )
                ).result()
        except Exception as exc:
            message = self._redact(str(exc))
            raise RuntimeError(f"Factory Droid SDK error for model {resolved_model}: {message}") from None

        if not result.success:
            error = getattr(getattr(result, "error", None), "message", None)
            detail = self._redact(str(error or result.subtype))
            raise RuntimeError(f"Factory Droid SDK error for model {resolved_model}: {detail}")

        usage = self._extract_usage(getattr(result, "usage", None))
        return ModelResponse(
            content=result.text,
            usage=usage,
            model_name=resolved_model,
            friendly_name=self.FRIENDLY_NAME,
            provider=ProviderType.FACTORY,
            metadata={"finish_reason": result.subtype, "sdk": "droid-sdk"},
        )

    async def _run_droid(
        self,
        *,
        prompt: str,
        model_name: str,
        system_prompt: Optional[str],
        thinking_mode: Optional[str],
        images: Optional[list[str]],
        max_image_size_mb: float,
    ):
        try:
            from droid_sdk import Autonomy, Image, ReasoningEffort, Runtime, SessionConfig, run
        except ImportError as exc:
            raise RuntimeError("droid-sdk is not installed") from exc

        effort_name = self.THINKING_EFFORT.get(thinking_mode or "", "HIGH")
        attachments = []
        for image_path in images or []:
            try:
                image_bytes, media_type = validate_image(image_path, max_image_size_mb)
                attachments.append(Image.from_bytes(image_bytes, media_type=media_type))
            except ValueError as exc:
                logger.warning("Ignoring invalid Factory image input: %s", exc)

        runtime = Runtime(
            executable=get_env("PAL_DROID_EXECUTABLE") or "droid",
            env=self._droid_environment(),
        )
        config = SessionConfig(
            autonomy=Autonomy.OFF,
            auto_reject_permission_requests=True,
            disable_builtin_skills=True,
            restrict_tools=(),
            system_prompt=system_prompt or None,
        )
        return await run(
            prompt,
            model=model_name,
            reasoning_effort=getattr(ReasoningEffort, effort_name),
            images=attachments,
            timeout=self.REQUEST_TIMEOUT_SECONDS,
            config=config,
            runtime=runtime,
            api_key=None if get_env_bool("PAL_FACTORY_DROID_LOCAL_AUTH") else (self.api_key or None),
        )

    def _droid_environment(self) -> dict[str, str]:
        environment = {
            name: value
            for name, value in os.environ.items()
            if name in self.DROID_PROCESS_ENVIRONMENT or name.startswith("FACTORY_") or name.startswith("LC_")
        }
        if get_env_bool("PAL_FACTORY_DROID_LOCAL_AUTH"):
            environment.pop("FACTORY_API_KEY", None)
        return environment

    def _redact(self, message: str) -> str:
        return message.replace(self.api_key, "[REDACTED]") if self.api_key else message

    @staticmethod
    def _extract_usage(raw_usage) -> dict[str, int]:
        if raw_usage is None:
            return {}
        usage = {}
        for source, target in (
            ("input_tokens", "input_tokens"),
            ("output_tokens", "output_tokens"),
            ("thinking_tokens", "thinking_tokens"),
        ):
            value = getattr(raw_usage, source, None)
            if isinstance(value, int):
                usage[target] = value
        if "input_tokens" in usage and "output_tokens" in usage:
            usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        return usage

    def get_preferred_model(self, category: "ToolModelCategory", allowed_models: list[str]) -> Optional[str]:
        if self.PRIMARY_MODEL in allowed_models:
            return self.PRIMARY_MODEL
        return allowed_models[0] if allowed_models else None


FactoryModelProvider._ensure_registry()
