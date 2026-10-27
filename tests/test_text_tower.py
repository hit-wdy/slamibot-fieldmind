"""The text-only CLIP loader must be interchangeable with the full one.

These check the invariant that matters for correctness: dropping the visual tower
does not move the text vectors or the model fingerprint. They need the pinned
assets, so they are marked slow.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from slamibot_fieldmind.config import Config
from slamibot_fieldmind.encoder import AlignedEncoder

pytestmark = pytest.mark.slow

TEXTS = [
    "a person wearing a red jacket",
    "a yellow fire hydrant near the road",
    "an empty corridor with a door",
]


@pytest.fixture(scope="module")
def configs():
    # The default config targets cuda:0; on a host without a usable CUDA torch
    # the comparison still has to run, so resolve the device the same way the
    # deployment does.
    import torch

    base = Config.default()
    if not base.model.checkpoint_path.is_file():
        pytest.skip("RADIO checkpoint is absent; run scripts/fetch_assets.py first")
    if not torch.cuda.is_available():
        base.model.device = "cpu"
        base.model.text_encoder_device = "cpu"
        base.model.dtype = "float32"
        base.model.text_encoder_dtype = "float32"
    full = copy.deepcopy(base)
    full.model.text_encoder_load = "full"
    text_only = copy.deepcopy(base)
    text_only.model.text_encoder_load = "text_only"
    return full, text_only


def test_text_only_matches_full_text_vectors(configs):
    full_cfg, text_cfg = configs
    full = AlignedEncoder(full_cfg)
    text = AlignedEncoder(text_cfg)
    a = full.encode_texts(TEXTS)
    b = text.encode_texts(TEXTS)
    assert np.allclose(np.sum(a * b, axis=1), 1.0, atol=1e-6)
    assert np.abs(a - b).max() < 1e-6


def test_text_only_keeps_the_same_fingerprint(configs):
    full_cfg, text_cfg = configs
    assert AlignedEncoder(full_cfg).fingerprint() == AlignedEncoder(text_cfg).fingerprint()


def test_text_only_reports_the_adaptor_and_no_visual_tower(configs):
    _, text_cfg = configs
    encoder = AlignedEncoder(text_cfg)
    assert encoder.load_report.adaptor_classes == {"clip": "OpenCLIP_RADIO"}
    adaptor = encoder.model.adaptors["clip"]
    assert adaptor.oc_model.visual is None
    assert callable(adaptor.oc_model.encode_text)


def test_unknown_text_encoder_load_is_rejected():
    with pytest.raises(ValueError, match="text_encoder_load"):
        Config.from_dict({"model": {"text_encoder_load": "bogus"}})
