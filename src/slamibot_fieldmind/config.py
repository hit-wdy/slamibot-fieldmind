"""Configuration for the image/text encoder.

The defaults mirror the placeholders in RADIO_Camera_Text_Retrieval_Spec.md
section 10, filled in with the values verified in phase A. Anything still
unverified is left as `None` and resolved from the loaded model, never guessed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import yaml

from .paths import ASSETS_MANIFEST, CONFIGS_DIR, PROJECT_ROOT

# Values verified against the pinned checkpoint and the official RESOURCE_MAP.
VERIFIED_VERSION = "radio_v2.5-b"
VERIFIED_ADAPTOR = "clip"
VERIFIED_SOURCE_COMMIT = "4ffaf15d0ff48197c194e306772bdb5fb92db230"
VERIFIED_CHECKPOINT_ASSET = "radio_v2.5_b"

DTYPES = ("auto", "float32", "bfloat16", "float16")


def asset_entry(name: str) -> dict[str, Any]:
    manifest = json.loads(ASSETS_MANIFEST.read_text(encoding="utf-8"))
    for asset in manifest["assets"]:
        if asset["name"] == name:
            return asset
    raise KeyError(f"asset {name!r} is not declared in {ASSETS_MANIFEST}")


def resolve_path(path: str | Path) -> Path:
    """Resolve a config path against the project root, not the CWD."""
    path = Path(path)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


@dataclass
class ModelConfig:
    """How the RADIO image backbone and its image/text adaptor are loaded."""

    version: str = VERIFIED_VERSION
    adaptor: str = VERIFIED_ADAPTOR
    backend: str = "radio"
    checkpoint: str = "models/hub/checkpoints/radio-v2.5-b_half.pth.tar"
    checkpoint_sha256: str = ""
    source_commit: str = VERIFIED_SOURCE_COMMIT

    device: str = "cuda:0"
    # The CLIP text tower is only needed for text encoding (step 2). Keeping it
    # off the GPU leaves the 4 GB of the RTX 3050 Ti for the image backbone.
    text_encoder_device: str = "cpu"
    dtype: str = "auto"
    text_encoder_dtype: str = "float32"
    # `text_only` builds just the paired CLIP text tower instead of the full CLIP
    # model. RADIO's adaptor deletes the visual tower immediately after building
    # it, so `full` only adds cold-start work; keep it for A/B verification.
    text_encoder_load: str = "text_only"

    # `[h, w]` selects center-crop mode, `[shortest_edge]` keeps the aspect
    # ratio. `resize_multiple: null` means "use the model's min_resolution_step".
    input_resolution: Sequence[int] = field(default_factory=lambda: [768, 768])
    resize_multiple: int | None = None
    max_dim: bool = False
    batch_size: int = 1

    def __post_init__(self) -> None:
        if self.backend != "radio":
            raise ValueError(
                f"model.backend {self.backend!r} is not implemented; only 'radio' has a verified "
                "image/text adaptor pairing (spec 2.1)"
            )
        if self.dtype not in DTYPES:
            raise ValueError(f"model.dtype must be one of {DTYPES}, got {self.dtype!r}")
        if self.text_encoder_dtype not in DTYPES[1:]:
            raise ValueError(
                f"model.text_encoder_dtype must be one of {DTYPES[1:]}, got {self.text_encoder_dtype!r}"
            )
        if self.text_encoder_load not in ("text_only", "full"):
            raise ValueError(
                f"model.text_encoder_load must be 'text_only' or 'full', got {self.text_encoder_load!r}"
            )
        resolution = [int(v) for v in (self.input_resolution or [])]
        if len(resolution) not in (1, 2) or any(v <= 0 for v in resolution):
            raise ValueError(
                "model.input_resolution must be [shortest_edge] or [height, width] of positive ints, "
                f"got {list(self.input_resolution)!r}"
            )
        self.input_resolution = resolution
        self.checkpoint_path: Path = resolve_path(self.checkpoint)
        if not self.checkpoint_sha256:
            self.checkpoint_sha256 = asset_entry(VERIFIED_CHECKPOINT_ASSET)["sha256"]


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)

    _SECTIONS = ("model",)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "Config":
        raw = raw or {}
        unknown = set(raw) - set(cls._SECTIONS)
        if unknown:
            raise ValueError(f"unknown config section(s): {', '.join(sorted(unknown))}")
        model_raw = dict(raw.get("model") or {})
        known_model_keys = set(ModelConfig.__dataclass_fields__) - {"checkpoint_path"}
        unknown_model = set(model_raw) - known_model_keys
        if unknown_model:
            raise ValueError(f"unknown model config key(s): {', '.join(sorted(unknown_model))}")
        return cls(model=ModelConfig(**model_raw))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def default(cls) -> "Config":
        return cls.from_yaml(CONFIGS_DIR / "default.yaml")
