"""Security and precedence tests for the external PAL credential config."""

import os
from pathlib import Path

import pytest

import utils.env as env_config

PROVIDER_KEYS = {
    "OPENAI_API_KEY": "openai-sentinel",
    "GEMINI_API_KEY": "gemini-sentinel",
    "ANTHROPIC_API_KEY": "anthropic-sentinel",
    "XAI_API_KEY": "xai-sentinel",
    "FACTORY_API_KEY": "factory-sentinel",
    "PAL_FACTORY_DROID_USE_LOCAL_LOGIN": "true",
}
MODEL_ALLOWLISTS = {
    "OPENAI_ALLOWED_MODELS": "gpt-5.6-sol",
    "GOOGLE_ALLOWED_MODELS": "gemini-3.1-pro-preview",
    "ANTHROPIC_ALLOWED_MODELS": "claude-fable-5",
    "XAI_ALLOWED_MODELS": "grok-4.6",
    "FACTORY_ALLOWED_MODELS": "deepseek-v4-pro",
}


@pytest.fixture
def external_config(monkeypatch, tmp_path):
    config_home = tmp_path / "config-home"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    monkeypatch.setattr(env_config, "_ENV_PATH", tmp_path / "missing-legacy-dotenv")
    for name in PROVIDER_KEYS | MODEL_ALLOWLISTS:
        monkeypatch.delenv(name, raising=False)

    path = config_home / "pal-mcp" / "config"
    path.parent.mkdir(parents=True, mode=0o700)
    try:
        yield path
    finally:
        env_config.reload_env({"PAL_MCP_FORCE_ENV_OVERRIDE": "false"})


def write_config(path: Path, values: dict[str, str], *, mode: int = 0o600) -> None:
    path.write_text("".join(f"{name}={value}\n" for name, value in values.items()))
    path.chmod(mode)


def test_external_config_loads_exact_provider_keys_and_model_allowlists(external_config):
    values = PROVIDER_KEYS | MODEL_ALLOWLISTS
    write_config(external_config, values)

    env_config.reload_env()

    assert env_config.credential_config_path() == external_config
    assert {name: env_config.get_env(name) for name in values} == values
    assert all(env_config.get_all_env()[name] is None for name in values)


def test_ambient_environment_takes_precedence(external_config, monkeypatch):
    write_config(external_config, {"OPENAI_API_KEY": "file-sentinel"})
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-sentinel")

    env_config.reload_env()

    assert env_config.get_env("OPENAI_API_KEY") == "ambient-sentinel"


def test_external_config_takes_precedence_over_legacy_dotenv(external_config, monkeypatch, tmp_path):
    legacy = tmp_path / "legacy.env"
    legacy.write_text("OPENAI_API_KEY=legacy-sentinel\nLEGACY_ONLY_SETTING=legacy-model\n")
    monkeypatch.delenv("LEGACY_ONLY_SETTING", raising=False)
    monkeypatch.setattr(env_config, "_ENV_PATH", legacy)
    write_config(external_config, {"OPENAI_API_KEY": "external-sentinel"})

    env_config.reload_env()

    assert env_config.get_env("OPENAI_API_KEY") == "external-sentinel"
    assert "OPENAI_API_KEY" not in os.environ
    assert os.environ["LEGACY_ONLY_SETTING"] == "legacy-model"


def test_relative_xdg_config_home_is_rejected(monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", "relative-config")

    with pytest.raises(RuntimeError, match="must be absolute"):
        env_config.credential_config_path()


def test_matching_quotes_are_removed_without_shell_evaluation(external_config):
    external_config.write_text('OPENAI_API_KEY="quoted-sentinel"\n')
    external_config.chmod(0o600)

    env_config.reload_env()

    assert env_config.get_env("OPENAI_API_KEY") == "quoted-sentinel"


def test_unknown_setting_is_rejected_without_echoing_value(external_config):
    secret = "must-not-appear"
    write_config(external_config, {"UNSUPPORTED_SETTING": secret})

    with pytest.raises(RuntimeError) as error:
        env_config.reload_env()

    assert secret not in str(error.value)


def test_duplicate_setting_is_rejected(external_config):
    external_config.write_text("OPENAI_API_KEY=first\nOPENAI_API_KEY=second\n")
    external_config.chmod(0o600)

    with pytest.raises(RuntimeError, match="repeats a setting"):
        env_config.reload_env()


def test_group_or_world_readable_file_is_rejected(external_config):
    write_config(external_config, PROVIDER_KEYS, mode=0o644)

    with pytest.raises(RuntimeError, match="mode must be 0600"):
        env_config.reload_env()


def test_symlink_file_is_rejected(external_config, tmp_path):
    target = tmp_path / "credential-target"
    write_config(target, PROVIDER_KEYS)
    external_config.symlink_to(target)

    with pytest.raises(RuntimeError, match="non-symlink"):
        env_config.reload_env()


def test_group_writable_parent_directory_is_rejected(external_config):
    write_config(external_config, PROVIDER_KEYS)
    external_config.parent.chmod(0o770)

    with pytest.raises(RuntimeError, match="directory must not be group- or world-writable"):
        env_config.reload_env()


def test_missing_external_config_preserves_regular_environment(external_config, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-sentinel")

    env_config.reload_env()

    assert not external_config.exists()
    assert env_config.get_env("OPENAI_API_KEY") == "ambient-sentinel"
