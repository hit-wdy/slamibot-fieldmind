"""Project paths, resolved from this file's location rather than the CWD."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RADIO_SOURCE = PROJECT_ROOT / "third_party" / "RADIO"
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_MANIFEST = MODELS_DIR / "assets.json"
TORCH_HUB_DIR = MODELS_DIR / "hub"
HF_HOME = MODELS_DIR / "hf"
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
