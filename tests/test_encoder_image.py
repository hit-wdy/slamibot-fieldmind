"""Image embedding contract: the startup checks the spec demands in section 2.2.

These load the real checkpoint, so they are marked slow. They assert wiring and
shape, not accuracy: passing them says nothing about retrieval quality.
"""

from __future__ import annotations

import copy
import dataclasses
import numpy as np
import pytest

from slamibot_fieldmind import AlignedEncoder, Config
from slamibot_fieldmind.encoder import _validate_embeddings, compute_fingerprint
from slamibot_fieldmind.preprocess import prepare_image

from conftest import camera_frame, distinct_frames

pytestmark = pytest.mark.slow


def test_embedding_shape_dtype_and_norm(encoder: AlignedEncoder):
    images = [camera_frame(seed) for seed in range(3)]
    embeddings = encoder.encode_images(images)

    assert embeddings.shape == (3, encoder.embedding_dim)
    assert embeddings.dtype == np.float32
    assert np.isfinite(embeddings).all()
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)
    assert np.abs(embeddings).max() > 0.0


def test_single_image_and_list_agree(encoder: AlignedEncoder):
    image = camera_frame(7)
    as_list = encoder.encode_images([image])
    # A bare array must not be mistaken for a batch of rows.
    direct = encoder.encode_images(image)
    assert as_list.shape == direct.shape == (1, encoder.embedding_dim)
    assert np.allclose(as_list, direct, atol=1e-6)


def test_batching_does_not_change_results(encoder: AlignedEncoder):
    images = [camera_frame(seed) for seed in range(3)]
    together = encoder.encode_images(images)
    separately = np.concatenate([encoder.encode_images([image]) for image in images], axis=0)
    assert np.allclose(together, separately, atol=1e-5)


def test_encoding_is_deterministic(encoder: AlignedEncoder):
    image = camera_frame(11)
    assert np.array_equal(encoder.encode_images([image]), encoder.encode_images([image]))


def test_distinct_images_do_not_collapse(encoder: AlignedEncoder):
    """A CLIP space compresses unrelated inputs, but not to a single point."""
    frames = distinct_frames()
    embeddings = encoder.encode_images(frames)
    similarity = embeddings @ embeddings.T
    off_diagonal = similarity[~np.eye(len(frames), dtype=bool)]
    assert similarity.diagonal().min() > 0.9999
    assert off_diagonal.max() < 0.99, "unrelated frames produced nearly identical embeddings"
    assert off_diagonal.min() < 0.95, "every frame mapped to nearly the same vector"


def test_fingerprint_is_stable(encoder: AlignedEncoder):
    assert encoder.fingerprint() == encoder.fingerprint()
    assert len(encoder.fingerprint()) == 64
    assert len(encoder.fingerprint_short()) == 16


@pytest.mark.parametrize(
    "change, value",
    [
        ("input_resolution", [512, 512]),
        ("max_dim", True),
        ("checkpoint_sha256", "0" * 64),
    ],
)
def test_fingerprint_tracks_the_vector_space(encoder: AlignedEncoder, change: str, value):
    assert compute_fingerprint(**_variant(encoder, **{change: value})) != encoder.fingerprint()


def test_fingerprint_tracks_the_run_dtype(encoder: AlignedEncoder):
    current = encoder.load_report.model_dtype
    other = "float16" if current.endswith("float32") else "float32"
    assert compute_fingerprint(**_variant(encoder, dtype=other)) != encoder.fingerprint()


def test_fingerprint_ignores_unrelated_settings(encoder: AlignedEncoder):
    other = _variant(encoder, device="cpu", batch_size=4, text_encoder_device="cuda:0")
    assert compute_fingerprint(**other) == encoder.fingerprint()


def test_unit_norm_violations_are_rejected():
    with pytest.raises(RuntimeError, match="L2 normalised"):
        _validate_embeddings(np.full((2, 4), 2.0, dtype=np.float32), 4)
    with pytest.raises(RuntimeError, match="expected"):
        _validate_embeddings(np.ones((2, 4), dtype=np.float32), 5)
    with pytest.raises(RuntimeError, match="all-zero"):
        _validate_embeddings(np.zeros((2, 4), dtype=np.float32), 4)
    with pytest.raises(RuntimeError, match="non-finite"):
        bad = np.full((2, 4), np.nan, dtype=np.float32)
        _validate_embeddings(bad, 4)


def test_prepared_shape_is_a_multiple_of_the_patch_size(encoder: AlignedEncoder):
    tensor = prepare_image(
        camera_frame(3), encoder.config.model.input_resolution, encoder.resize_multiple
    )
    assert tensor.shape[-2] % encoder.resize_multiple == 0
    assert tensor.shape[-1] % encoder.resize_multiple == 0


def test_non_multiple_center_crop_resolution_is_rejected(encoder: AlignedEncoder):
    config = copy.deepcopy(encoder.config)
    config.model.input_resolution = [700, 700]
    config.model.__post_init__()
    with pytest.raises(ValueError, match="multiple of"):
        AlignedEncoder(config)


def _variant(encoder: AlignedEncoder, **overrides) -> dict:
    """Rebuild the fingerprint inputs with a few config values changed."""
    fields = {field.name for field in dataclasses.fields(encoder.config.model)}
    model_config = dataclasses.replace(
        encoder.config.model, **{k: v for k, v in overrides.items() if k in fields}
    )
    report = encoder.load_report
    if "dtype" in overrides:
        report = dataclasses.replace(report, model_dtype=f"torch.{overrides['dtype']}")
    return dict(
        model_config=model_config,
        report=report,
        embedding_dim=encoder.embedding_dim,
        resize_multiple=encoder.resize_multiple,
    )
