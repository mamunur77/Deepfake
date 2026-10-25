"""
src.utils.config  ·  YAML configuration loader & merger.
========================================================
Loads configs/default.yaml (or any YAML path) into a nested namespace.
CLI flags can override individual keys.

Usage:
    from src.utils.config import load_config
    cfg = load_config("configs/default.yaml")
    print(cfg.model.backbone)
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from types import SimpleNamespace

import yaml


def _dict_to_namespace(d: dict) -> SimpleNamespace:
    """Recursively convert a nested dict into SimpleNamespace for dot-access."""
    ns = SimpleNamespace()
    for key, val in d.items():
        if isinstance(val, dict):
            setattr(ns, key, _dict_to_namespace(val))
        else:
            setattr(ns, key, val)
    return ns


def _namespace_to_dict(ns: SimpleNamespace) -> dict:
    """Recursively convert a SimpleNamespace back into a plain dict."""
    d = {}
    for key, val in vars(ns).items():
        if isinstance(val, SimpleNamespace):
            d[key] = _namespace_to_dict(val)
        else:
            d[key] = val
    return d


def load_config(path: str | Path) -> SimpleNamespace:
    """
    Load a YAML config file and return a dot-accessible namespace.

    Parameters
    ----------
    path : str | Path
        Path to the YAML configuration file.

    Returns
    -------
    SimpleNamespace
        Nested namespace with all config values.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return _dict_to_namespace(raw)


def save_config(cfg: SimpleNamespace, path: str | Path) -> None:
    """Save a config namespace back to a YAML file (for checkpoint reproducibility)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(_namespace_to_dict(cfg), f, default_flow_style=False, sort_keys=False)


def apply_debug_overrides(cfg: SimpleNamespace) -> SimpleNamespace:
    """
    If cfg.debug.enabled is True, override training hyper-parameters
    with safe laptop-friendly values.
    """
    if getattr(cfg.debug, "enabled", False):
        cfg.training.epochs = cfg.debug.epochs
        cfg.training.batch_size = cfg.debug.batch_size
        print(
            f"[DEBUG MODE] epochs={cfg.debug.epochs}, "
            f"batch_size={cfg.debug.batch_size}, "
            f"max_samples={cfg.debug.max_samples}"
        )
    return cfg


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add common CLI arguments for config path and debug toggle."""
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug mode (50 samples, 1 epoch, batch 4).",
    )
    return parser
