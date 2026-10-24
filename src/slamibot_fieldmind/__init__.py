"""SLAMIBOT FieldMind: visual memory and retrieval for mobile robots.

Model-backed exports are imported on demand so MCP proxies stay lightweight.
"""

from __future__ import annotations

from importlib import import_module

# A stdio proxy must not import PyTorch just to forward JSON and JPEG content.
_MODULES = {
    "LoadReport": "backbone", "load_radio_model": "backbone",
    "model_facts": "backbone", "verify_checkpoint": "backbone",
    "Config": "config", "ModelConfig": "config",
    "CODE_VERSION": "encoder", "AlignedEncoder": "encoder",
    "compute_fingerprint": "encoder", "describe": "encoder",
    "prepare_batch": "preprocess", "prepare_image": "preprocess",
}


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(name)
    value = getattr(import_module("." + _MODULES[name], __name__), name)
    globals()[name] = value
    return value

__all__ = [
    "AlignedEncoder",
    "CODE_VERSION",
    "Config",
    "LoadReport",
    "ModelConfig",
    "compute_fingerprint",
    "describe",
    "load_radio_model",
    "model_facts",
    "prepare_batch",
    "prepare_image",
    "verify_checkpoint",
]
