"""Centralized environment variable access for PAL MCP Server."""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path

try:
    from dotenv import dotenv_values, load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    dotenv_values = None  # type: ignore[assignment]
    load_dotenv = None  # type: ignore[assignment]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _PROJECT_ROOT / ".env"
_CONFIG_ENV_NAMES = frozenset(
    {
        "ANTHROPIC_ALLOWED_MODELS",
        "ANTHROPIC_API_KEY",
        "FACTORY_ALLOWED_MODELS",
        "FACTORY_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_ALLOWED_MODELS",
        "OPENAI_ALLOWED_MODELS",
        "OPENAI_API_KEY",
        "XAI_ALLOWED_MODELS",
        "XAI_API_KEY",
    }
)
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
_MAX_CONFIG_BYTES = 64 * 1024

_DOTENV_VALUES: dict[str, str | None] = {}
_FORCE_ENV_OVERRIDE = False


def credential_config_path() -> Path:
    """Return the mutable, cross-platform PAL credential configuration path."""
    xdg_config_home = os.getenv("XDG_CONFIG_HOME")
    if xdg_config_home:
        config_home = Path(xdg_config_home).expanduser()
        if not config_home.is_absolute():
            raise RuntimeError("XDG_CONFIG_HOME must be absolute for PAL credential config")
    else:
        config_home = Path.home() / ".config"
    return config_home / "pal-mcp" / "config"


def _validate_config_owner(metadata: os.stat_result, description: str) -> None:
    if hasattr(os, "getuid") and metadata.st_uid != os.getuid():
        raise RuntimeError(f"PAL credential config {description} must be owned by the current user")


def _parse_credential_config(payload: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, line in enumerate(payload.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        name, separator, raw_value = stripped.partition("=")
        if not separator or name != name.strip() or _ENV_NAME.fullmatch(name) is None:
            raise RuntimeError(f"PAL credential config has invalid syntax at line {line_number}")
        if name not in _CONFIG_ENV_NAMES:
            raise RuntimeError(f"PAL credential config has an unsupported setting at line {line_number}")
        if name in values:
            raise RuntimeError(f"PAL credential config repeats a setting at line {line_number}")

        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        elif value.startswith(("'", '"')) or value.endswith(("'", '"')):
            raise RuntimeError(f"PAL credential config has an unmatched quote at line {line_number}")
        if not value or "\x00" in value:
            raise RuntimeError(f"PAL credential config has an empty or invalid value at line {line_number}")
        values[name] = value

    return values


def _read_credential_config() -> dict[str, str]:
    path = credential_config_path()
    try:
        path_metadata = os.lstat(path)
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise RuntimeError("PAL credential config metadata is unavailable") from exc

    if stat.S_ISLNK(path_metadata.st_mode) or not stat.S_ISREG(path_metadata.st_mode):
        raise RuntimeError("PAL credential config must be a regular, non-symlink file")
    _validate_config_owner(path_metadata, "file")

    if os.name == "posix":
        if stat.S_IMODE(path_metadata.st_mode) != 0o600:
            raise RuntimeError("PAL credential config file mode must be 0600")
        if path_metadata.st_nlink != 1:
            raise RuntimeError("PAL credential config must have exactly one hard link")

        try:
            parent_metadata = os.lstat(path.parent)
        except OSError as exc:
            raise RuntimeError("PAL credential config directory metadata is unavailable") from exc
        if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(parent_metadata.st_mode):
            raise RuntimeError("PAL credential config directory must be a regular directory")
        _validate_config_owner(parent_metadata, "directory")
        if stat.S_IMODE(parent_metadata.st_mode) & 0o022:
            raise RuntimeError("PAL credential config directory must not be group- or world-writable")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError("PAL credential config could not be opened safely") from exc

    try:
        opened_metadata = os.fstat(descriptor)
        if not stat.S_ISREG(opened_metadata.st_mode):
            raise RuntimeError("PAL credential config changed before it was opened")
        if (opened_metadata.st_dev, opened_metadata.st_ino) != (path_metadata.st_dev, path_metadata.st_ino):
            raise RuntimeError("PAL credential config changed before it was opened")
        _validate_config_owner(opened_metadata, "file")
        if os.name == "posix" and (stat.S_IMODE(opened_metadata.st_mode) != 0o600 or opened_metadata.st_nlink != 1):
            raise RuntimeError("PAL credential config changed before it was opened")
        if opened_metadata.st_size > _MAX_CONFIG_BYTES:
            raise RuntimeError("PAL credential config exceeds the size limit")
        with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
            descriptor = -1
            payload = stream.read(_MAX_CONFIG_BYTES + 1)
    except UnicodeDecodeError as exc:
        raise RuntimeError("PAL credential config must be valid UTF-8") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    if len(payload.encode("utf-8")) > _MAX_CONFIG_BYTES:
        raise RuntimeError("PAL credential config exceeds the size limit")
    return _parse_credential_config(payload)


def _read_dotenv_values() -> dict[str, str | None]:
    if dotenv_values is not None and _ENV_PATH.exists():
        loaded = dotenv_values(_ENV_PATH, interpolate=False)
        return dict(loaded)
    return {}


def _compute_force_override(values: Mapping[str, str | None]) -> bool:
    raw = (values.get("PAL_MCP_FORCE_ENV_OVERRIDE") or "false").strip().lower()
    return raw == "true"


def _apply_legacy_environment(legacy_values: Mapping[str, str | None], credential_values: Mapping[str, str]) -> None:
    for name, value in legacy_values.items():
        if value is None or name in credential_values:
            continue
        if _FORCE_ENV_OVERRIDE or name not in os.environ:
            os.environ[name] = value


def reload_env(dotenv_mapping: Mapping[str, str | None] | None = None) -> None:
    """Reload legacy dotenv values and the strict external PAL configuration."""
    global _DOTENV_VALUES, _FORCE_ENV_OVERRIDE

    if dotenv_mapping is not None:
        _DOTENV_VALUES = dict(dotenv_mapping)
        _FORCE_ENV_OVERRIDE = _compute_force_override(_DOTENV_VALUES)
        return

    legacy_values = _read_dotenv_values()
    credential_values = _read_credential_config()
    _DOTENV_VALUES = legacy_values | credential_values
    _FORCE_ENV_OVERRIDE = _compute_force_override(legacy_values)

    if credential_values:
        _apply_legacy_environment(legacy_values, credential_values)
    elif load_dotenv is not None and _ENV_PATH.exists():
        load_dotenv(dotenv_path=_ENV_PATH, override=_FORCE_ENV_OVERRIDE)


reload_env()


def env_override_enabled() -> bool:
    """Return True when legacy PAL_MCP_FORCE_ENV_OVERRIDE is enabled."""
    return _FORCE_ENV_OVERRIDE


def get_env(key: str, default: str | None = None) -> str | None:
    """Retrieve an environment value with ambient environment taking precedence."""
    if env_override_enabled():
        if key in _DOTENV_VALUES:
            value = _DOTENV_VALUES[key]
            return value if value is not None else default
        return default

    if key in os.environ:
        return os.environ[key]
    value = _DOTENV_VALUES.get(key)
    return value if value is not None else default


def get_env_bool(key: str, default: bool = False) -> bool:
    """Boolean helper that respects environment precedence."""
    raw_default = "true" if default else "false"
    raw_value = get_env(key, raw_default)
    return (raw_value or raw_default).strip().lower() == "true"


def get_all_env() -> dict[str, str | None]:
    """Expose loaded diagnostics while redacting external credential settings."""
    return {name: None if name in _CONFIG_ENV_NAMES else value for name, value in _DOTENV_VALUES.items()}


@contextmanager
def suppress_env_vars(*names: str):
    """Temporarily remove environment variables during the context."""
    removed: dict[str, str] = {}
    try:
        for name in names:
            if not name:
                continue
            if name in os.environ:
                removed[name] = os.environ[name]
                del os.environ[name]
        yield
    finally:
        for name, value in removed.items():
            os.environ[name] = value
