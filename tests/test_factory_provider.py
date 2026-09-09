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

    @patch.dict(
        os.environ,
        {
            "HOME": "/tmp/factory-home",
            "FACTORY_API_KEY": "ambient-decoy",
            "PAL_FACTORY_DROID_USE_LOCAL_LOGIN": "true",
        },
        clear=True,
    )
    @patch("droid_sdk.transport.ProcessTransport")
    @patch("droid_sdk.run", new_callable=AsyncMock)
    def test_local_droid_auth_ignores_explicit_api_key(self, sdk_run, transport_class):
        transport = transport_class.return_value
        transport.connect = AsyncMock()
        observed_keys = []

        async def run_with_environment_check(*_args, **_kwargs):
            observed_keys.append(os.environ.get("FACTORY_API_KEY"))
            return successful_result()

        sdk_run.side_effect = run_with_environment_check

        FactoryModelProvider("argument-decoy").generate_content(prompt="hello", model_name="deepseek-v4-pro")

        call = sdk_run.await_args.kwargs
        assert call["api_key"] is None
        assert dict(call["runtime"].env) == {}
        assert call["runtime"].transport is transport
        transport.connect.assert_awaited_once()
        transport_class.assert_called_once_with(
            exec_path="droid",
            cwd=os.getcwd(),
            env={"HOME": "/tmp/factory-home", "FACTORY_API_KEY": ""},
        )
        assert observed_keys == ["ambient-decoy"]

    @patch.dict(
        os.environ,
        {
            "PAL_FACTORY_DROID_USE_LOCAL_LOGIN": "yes",
            "PAL_DROID_EXECUTABLE": "/managed/droid",
        },
        clear=True,
    )
    @patch("shutil.which", return_value="/managed/droid")
    def test_configure_providers_accepts_valid_managed_droid(self, which):
        import server

        ModelProviderRegistry.reset_for_testing()
        try:
            server.configure_providers()
            assert ProviderType.FACTORY in ModelProviderRegistry.get_available_providers()
            which.assert_called_once_with("/managed/droid")
        finally:
            ModelProviderRegistry.reset_for_testing()

    @patch.dict(
        os.environ,
        {
            "PAL_FACTORY_DROID_USE_LOCAL_LOGIN": "true",
            "PAL_DROID_EXECUTABLE": "/missing/droid",
        },
        clear=True,
    )
    @patch("shutil.which", return_value=None)
    def test_configure_providers_rejects_missing_managed_droid(self, which):
        import server

        ModelProviderRegistry.reset_for_testing()
        try:
            with pytest.raises(RuntimeError, match="Droid executable is unavailable"):
                server.configure_providers()
            which.assert_called_once_with("/missing/droid")
        finally:
            ModelProviderRegistry.reset_for_testing()

    @patch("droid_sdk.run", new_callable=AsyncMock)
    def test_unsupported_generation_inputs_fail_before_sdk(self, sdk_run):
        provider = FactoryModelProvider("factory-secret")
        with pytest.raises(ValueError, match="does not support max_output_tokens"):
            provider.generate_content(
                prompt="hello",
                model_name="deepseek-v4-pro",
                max_output_tokens=128,
            )
        with pytest.raises(ValueError, match="does not support images"):
            provider.generate_content(
                prompt="hello",
                model_name="deepseek-v4-pro",
                images=["/tmp/image.png"],
            )
        sdk_run.assert_not_awaited()

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
