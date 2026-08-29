"""Registry loader for Factory model capabilities."""

from __future__ import annotations

from ..shared import ProviderType
from .base import CapabilityModelRegistry


class FactoryModelRegistry(CapabilityModelRegistry):
    """Capability registry backed by ``conf/factory_models.json``."""

    def __init__(self, config_path: str | None = None) -> None:
        super().__init__(
            env_var_name="FACTORY_MODELS_CONFIG_PATH",
            default_filename="factory_models.json",
            provider=ProviderType.FACTORY,
            friendly_prefix="Factory ({model})",
            config_path=config_path,
        )
