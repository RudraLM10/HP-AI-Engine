"""
Configuration loader for HP AI Engine.

Loads YAML configuration files and merges with environment variable overrides.
Environment variables follow the pattern: HP_AI__<SECTION>__<KEY>=<VALUE>
e.g. HP_AI__GCN__HIDDEN_DIM=128 overrides gcn.hidden_dim in model_config.yaml.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


# Default config directory relative to project root
_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"

_ENV_PREFIX = "HP_AI__"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_env_overrides(config: dict) -> dict:
    """
    Apply environment variable overrides to config dict.

    Environment variables matching HP_AI__SECTION__KEY=VALUE are parsed
    and applied to the corresponding nested config key (case-insensitive).
    Numeric strings are auto-cast to int or float.
    """
    for env_key, env_value in os.environ.items():
        if not env_key.startswith(_ENV_PREFIX):
            continue
        parts = env_key[len(_ENV_PREFIX):].lower().split("__")
        if len(parts) < 2:
            continue

        # Navigate to the correct nesting level
        target = config
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]

        # Auto-cast value
        cast_value: Any = env_value
        try:
            cast_value = int(env_value)
        except ValueError:
            try:
                cast_value = float(env_value)
            except ValueError:
                if env_value.lower() in ("true", "false"):
                    cast_value = env_value.lower() == "true"

        target[parts[-1]] = cast_value

    return config


def _dict_to_namespace(d: dict) -> SimpleNamespace:
    """Recursively convert a dict to a SimpleNamespace for dot-access."""
    ns = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(ns, key, _dict_to_namespace(value))
        elif isinstance(value, list):
            setattr(ns, key, [
                _dict_to_namespace(item) if isinstance(item, dict) else item
                for item in value
            ])
        else:
            setattr(ns, key, value)
    return ns


def load_yaml(path: str | Path) -> dict:
    """Load a single YAML file and return as dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_config(
    config_dir: str | Path | None = None,
    files: list[str] | None = None,
    apply_env: bool = True,
) -> SimpleNamespace:
    """
    Load and merge all YAML config files from a directory.

    Args:
        config_dir: Path to config directory. Defaults to project's configs/ dir.
        files: Specific filenames to load. Defaults to all .yaml files.
        apply_env: Whether to apply HP_AI__* env-var overrides.

    Returns:
        SimpleNamespace with dot-accessible nested config values.
    """
    config_dir = Path(config_dir) if config_dir else _DEFAULT_CONFIG_DIR

    if files is None:
        yaml_files = sorted(config_dir.glob("*.yaml"))
    else:
        yaml_files = [config_dir / f for f in files]

    merged: dict = {}
    for yaml_file in yaml_files:
        if yaml_file.exists():
            # Use the filename stem as a top-level key
            stem = yaml_file.stem.replace("_config", "")
            data = load_yaml(yaml_file)
            merged[stem] = data

    if apply_env:
        merged = _apply_env_overrides(merged)

    return _dict_to_namespace(merged)


def load_single_config(filename: str, config_dir: str | Path | None = None) -> SimpleNamespace:
    """
    Load a single config file and return as SimpleNamespace.

    Args:
        filename: Name of the YAML file (e.g. 'model_config.yaml').
        config_dir: Path to config directory.

    Returns:
        SimpleNamespace with dot-accessible config values.
    """
    config_dir = Path(config_dir) if config_dir else _DEFAULT_CONFIG_DIR
    data = load_yaml(config_dir / filename)
    data = _apply_env_overrides(data)
    return _dict_to_namespace(data)
