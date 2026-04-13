"""
Shared YAML configuration utilities.

Provides a standardized way to load YAML config files and merge them
with CLI arguments, where CLI arguments take precedence over YAML values.

Usage in scripts:
    parser = argparse.ArgumentParser(...)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file')
    # ... add other arguments ...

    args = load_config_into_args(parser)
"""

import yaml
import argparse
from pathlib import Path
from typing import Optional


def load_config_into_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """
    Parse CLI args with optional YAML config file support.

    If --config is provided, loads the YAML file and uses its values as
    defaults. CLI arguments explicitly provided by the user override
    YAML values.

    Args:
        parser: ArgumentParser with --config and other arguments defined.

    Returns:
        Parsed argparse.Namespace with merged config.
    """
    # Save original required flags, then temporarily disable them
    # so parse_known_args() can extract --config without errors.
    required_actions = {}
    for action in parser._actions:
        if action.required:
            required_actions[action] = True
            action.required = False

    # First pass: extract --config path
    preliminary, _ = parser.parse_known_args()

    config_path = getattr(preliminary, 'config', None)
    if config_path is not None:
        yaml_config = load_yaml_config(config_path)
        if yaml_config:
            # Set YAML values as defaults (CLI args will override)
            parser.set_defaults(**yaml_config)

    # Restore required flags, but relax for args satisfied by YAML config
    for action, was_required in required_actions.items():
        if config_path is not None and yaml_config and action.dest in yaml_config:
            # YAML provides a value, so don't require CLI
            action.required = False
        else:
            action.required = was_required

    # Second pass: full parse with updated defaults
    args = parser.parse_args()
    return args


def load_yaml_config(config_path: str) -> dict:
    """Load a YAML config file and return as flat dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    if config is None:
        return {}

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping, got {type(config).__name__}")

    return config
