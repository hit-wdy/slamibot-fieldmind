"""Load only the paired CLIP text tower instead of the whole CLIP model.

RADIO's `clip` adaptor keeps only the text tower at runtime: `OpenCLIP_RADIO`
builds a full OpenCLIP model through `open_clip.create_model_from_pretrained`,
then immediately sets `oc_model.visual = None`. On a desktop GPU that wasted
allocation is hidden by the driver; on Orin NX the CPU and GPU share physical
memory, so building the 2.5 GB visual tower just to delete it roughly triples the
cold-start peak.

This module rebuilds the *same* text tower, loads only the non-`visual.* keys
from the pinned checkpoint, and exposes `encode_text`/`visual` with the same
surface `OpenCLIP_RADIO` expects. The tokenizer, text projection, normalisation
and weights are unchanged, so the vector space is identical; `backbone` verifies
that by comparing against the full load in tests.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_text_only_clip(
    model_name: str,
    pretrained: str,
    checkpoint_path: str | Path | None = None,
    dtype: torch.dtype | None = None,
) -> Any:
    """Return an OpenCLIP-compatible module with only the text tower loaded.

    `checkpoint_path` points at the already-downloaded `.bin`; when omitted the
    file is resolved from the OpenCLIP pretrained config (which honours
    `HF_ENDPOINT`/`HF_HUB_OFFLINE` through huggingface_hub).
    """
    import open_clip
    from open_clip.model import _build_text_tower, text_global_pool
    from open_clip.pretrained import get_pretrained_cfg

    config = open_clip.get_model_config(model_name)
    pretrained_cfg = get_pretrained_cfg(model_name, pretrained)
    if not config or not pretrained_cfg:
        raise RuntimeError(
            f"open_clip has no built-in config for {model_name!r}/{pretrained!r}"
        )
    quick_gelu = bool(pretrained_cfg.get("quick_gelu", config.get("quick_gelu", False)))

    model = _TextOnlyCLIP(
        embed_dim=int(config["embed_dim"]),
        text_cfg=config["text_cfg"],
        quick_gelu=quick_gelu,
        build_text_tower=_build_text_tower,
        text_global_pool=text_global_pool,
    )

    path = Path(checkpoint_path) if checkpoint_path else _resolve_checkpoint(
        model_name, pretrained
    )
    state = _load_text_state(path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # `logit_scale` belongs to the full CLIP head and is unused for text encoding.
    unexpected = [k for k in unexpected if k != "logit_scale"]
    if missing or unexpected:
        raise RuntimeError(
            f"text tower weights did not line up with {path.name}: "
            f"missing={missing[:6]} unexpected={unexpected[:6]}"
        )
    if dtype is not None:
        model.to(dtype=dtype)
    model.eval()
    model.requires_grad_(False)
    return model


def _resolve_checkpoint(model_name: str, pretrained: str) -> Path:
    from open_clip.pretrained import download_pretrained_from_hf, get_pretrained_cfg

    cfg = get_pretrained_cfg(model_name, pretrained)
    repo = (cfg.get("hf_hub") or "").rstrip("/")
    if not repo:
        raise RuntimeError(
            f"{model_name}/{pretrained} is not published on the Hugging Face Hub; "
            "set text_encoder_load: full to build it the original way"
        )
    return Path(download_pretrained_from_hf(repo))


def _load_text_state(path: Path) -> dict[str, torch.Tensor]:
    """Read the checkpoint and drop every `visual.*` tensor.

    `mmap=True` avoids materialising a second copy of the 3.95 GB file while the
    text weights are assigned; without it the load peak is roughly the file size
    again.
    """
    kwargs = dict(map_location="cpu", weights_only=True)
    try:
        state = torch.load(path, mmap=True, **kwargs)
    except (TypeError, RuntimeError):
        state = torch.load(path, **kwargs)
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    return {k: v for k, v in state.items() if not k.startswith("visual.")}


class _TextOnlyCLIP(nn.Module):
    """The text half of `open_clip.model.CLIP`, with the same encode path."""

    def __init__(
        self,
        embed_dim: int,
        text_cfg: Any,
        quick_gelu: bool,
        build_text_tower,
        text_global_pool,
    ) -> None:
        super().__init__()
        text = build_text_tower(embed_dim, text_cfg, quick_gelu, None)
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.text_pool_type = text.pool_type
        self.text_eos_id = text.eos_id
        self.register_buffer("attn_mask", text.attn_mask, persistent=False)
        # Kept as an attribute so the adaptor's `oc_model.visual = None` is a
        # no-op rather than an AttributeError.
        self.visual = None
        self._text_global_pool = text_global_pool

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()
        x = self.token_embedding(text).to(cast_dtype)
        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)
        x = self._text_global_pool(
            x, text, self.text_pool_type, eos_token_id=getattr(self, "text_eos_id", None)
        )
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                x = self.text_projection(x)
            else:
                x = x @ self.text_projection
        return F.normalize(x, dim=-1) if normalize else x


@contextlib.contextmanager
def text_only_open_clip(dtype: torch.dtype | None = None):
    """Patch `open_clip.create_model_from_pretrained` for the duration.

    RADIO's adaptor imports the package, not the function, so replacing the
    package attribute is enough. Everything else (tokenizer, adaptor wiring) is
    untouched.
    """
    import open_clip

    original = open_clip.create_model_from_pretrained

    def create_model_from_pretrained(model_name, pretrained=None, **kwargs):
        if not pretrained:
            # No tag means the caller wants a plain model; keep original semantics.
            return original(model_name, pretrained=pretrained, **kwargs)
        return build_text_only_clip(
            model_name,
            pretrained,
            checkpoint_path=kwargs.get("checkpoint_path"),
            dtype=dtype,
        )

    open_clip.create_model_from_pretrained = create_model_from_pretrained
    try:
        yield
    finally:
        open_clip.create_model_from_pretrained = original


def cached_paired_checkpoint(asset_name: str, repo_id: str, filename: str) -> Path | None:
    """Find a pinned CLIP checkpoint in the project HF cache without downloading."""
    from .paths import HF_HOME

    repo_dir = HF_HOME / "hub" / f"models--{repo_id.replace('/', '--')}"
    for candidate in repo_dir.glob(f"snapshots/*/{filename}"):
        if candidate.is_file():
            return candidate
    return None
