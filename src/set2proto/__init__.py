"""Set2Proto-Diffusion research MVP package."""

from .config import ConfigError, ResolvedConfig, load_config, validate_config

__all__ = [
    "ConfigError",
    "ResolvedConfig",
    "load_config",
    "validate_config",
]

__version__ = "0.1.0"
