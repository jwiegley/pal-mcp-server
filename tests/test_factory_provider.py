"""Tests for Factory models routed through Droid SDK."""

import os
import traceback
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from providers.factory import FactoryModelProvider
from providers.registry import ModelProviderRegistry
from providers.shared import ProviderType


def successful_result(text="Factory response"):
    return SimpleNamespace(
        success=True,
        subtype="success",
        text=text,
        error=None,
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, thinking_tokens=3),
    )


class TestFactoryProvider:
    def setup_method(self):
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None

    def teardown_method(self):
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None

    def test_exact_model_capabilities_and_fail_closed_resolution(self):
        provider = FactoryModelProvider("test-key")
        capabilities = provider.get_capabilities("deepseek-v4-pro")

        assert provider.get_provider_type() == ProviderType.FACTORY
        assert capabilities.model_name == "deepseek-v4-pro"
        assert capabilities.provider == ProviderType.FACTORY
        assert capabilities.context_window == 200_000
        assert capabilities.max_output_tokens == 16_384
        assert capabilities.default_reasoning_effort == "high"
        assert capabilities.supports_extended_thinking is True
        assert capabilities.supports_images is False
        assert capabilities.supports_temperature is False
        assert provider.list_models(respect_restrictions=False) == ["deepseek-v4-pro"]
        assert provider.validate_model_name("kimi-k3") is False

    @patch.dict(os.environ, {}, clear=True)
    def test_registry_routes_exact_model_without_requiring_environment_key(self):
        ModelProviderRegistry.reset_for_testing()
        try:
            ModelProviderRegistry.register_provider(ProviderType.FACTORY, FactoryModelProvider)
            provider = ModelProviderRegistry.get_provider_for_model("deepseek-v4-pro")

            assert isinstance(provider, FactoryModelProvider)
            assert provider.api_key == ""
            assert ModelProviderRegistry.get_provider_for_model("kimi-k3") is None
        finally:
            ModelProviderRegistry.reset_for_testing()

    @patch.dict(os.environ, {"FACTORY_ALLOWED_MODELS": "other-model"}, clear=True)
    def test_factory_allowlist_fails_closed(self):
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None
        provider = FactoryModelProvider("test-key")
        assert provider.validate_model_name("deepseek-v4-pro") is False

    @patch.dict(
        os.environ,
        {
            "HOME": "/tmp/factory-home",
            "PAL_DROID_EXECUTABLE": "/nix/store/test-droid/bin/droid",
            "FACTORY_API_KEY": "factory-secret",
            "FACTORY_OTEL_ENABLED": "false",
            "OPENAI_API_KEY": "must-not-reach-droid",
            "SSH_AUTH_SOCK": "/tmp/must-not-reach-droid",
        },
        clear=True,
    )
    @patch("droid_sdk.run", new_callable=AsyncMock)
    def test_exact_sdk_call_normalizes_response_and_filters_environment(self, sdk_run):
        sdk_run.return_value = successful_result()

        response = FactoryModelProvider("factory-secret").generate_content(
            prompt="Review this code",
            model_name="deepseek-v4-pro",
            system_prompt="Be precise",
            thinking_mode="max",
        )

        call = sdk_run.await_args.kwargs
        assert call["model"] == "deepseek-v4-pro"
        assert call["reasoning_effort"].value == "max"
        assert call["api_key"] == "factory-secret"
        assert call["runtime"].executable == Path("/nix/store/test-droid/bin/droid")
        assert call["runtime"].env["HOME"] == "/tmp/factory-home"
        assert call["runtime"].env["FACTORY_OTEL_ENABLED"] == "false"
        assert "OPENAI_API_KEY" not in call["runtime"].env
        assert "SSH_AUTH_SOCK" not in call["runtime"].env
        assert call["config"].system_prompt == "Be precise"
        assert call["config"].disable_builtin_skills is True
        assert call["config"].auto_reject_permission_requests is True
        assert call["config"].restrict_tools == frozenset()
        assert response.content == "Factory response"
        assert response.model_name == "deepseek-v4-pro"
        assert response.provider == ProviderType.FACTORY
        assert response.usage == {
            "input_tokens": 10,
            "output_tokens": 5,
            "thinking_tokens": 3,
            "total_tokens": 15,
        }

    @patch.dict(os.environ, {"HOME": "/tmp/factory-home"}, clear=True)
    @patch("droid_sdk.run", new_callable=AsyncMock)
    def test_local_droid_auth_state_is_preserved_without_api_key(self, sdk_run):
        sdk_run.return_value = successful_result()

        FactoryModelProvider().generate_content(prompt="hello", model_name="deepseek-v4-pro")

        call = sdk_run.await_args.kwargs
        assert call["api_key"] is None
        assert call["runtime"].env == {"HOME": "/tmp/factory-home"}

    @patch("droid_sdk.run", new_callable=AsyncMock)
    def test_invalid_model_never_reaches_sdk(self, sdk_run):
        with pytest.raises(ValueError, match="Unsupported model 'kimi-k3' for provider factory"):
            FactoryModelProvider("factory-secret").generate_content(prompt="hello", model_name="kimi-k3")

        sdk_run.assert_not_awaited()

    @patch("droid_sdk.run", new_callable=AsyncMock)
    def test_sdk_failure_is_redacted_and_never_falls_back(self, sdk_run):
        secret = "factory-" + "secret"
        sdk_run.side_effect = RuntimeError(f"authentication failed for {secret}")

        with pytest.raises(RuntimeError) as error:
            FactoryModelProvider(secret).generate_content(prompt="hello", model_name="deepseek-v4-pro")

        assert "[REDACTED]" in str(error.value)
        assert secret not in str(error.value)
        assert "deepseek-v4-pro" in str(error.value)
        assert secret not in "".join(traceback.format_exception(error.value))
