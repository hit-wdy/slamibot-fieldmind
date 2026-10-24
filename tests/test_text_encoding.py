"""Text encoding and the image-text alignment the spec asks for in section 2.2.

The ranking checks are falsifiable on synthetic frames whose content is known by
construction, so they run without any dataset. They prove the two sides share one
vector space; they are not an accuracy result.
"""

from __future__ import annotations

import numpy as np
import pytest

from slamibot_fieldmind import AlignedEncoder
from slamibot_fieldmind.paths import PROJECT_ROOT

from conftest import camera_frame, distinct_frames

pytestmark = pytest.mark.slow

CAT_PHOTO = PROJECT_ROOT / "data" / "test_assets" / "cat.png"


def test_text_embedding_shape_dtype_and_norm(encoder: AlignedEncoder):
    embeddings = encoder.encode_texts(["a fire truck", "a cat", "an empty street"])
    assert embeddings.shape == (3, encoder.embedding_dim)
    assert embeddings.dtype == np.float32
    assert np.isfinite(embeddings).all()
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)


def test_text_batching_does_not_change_results(encoder: AlignedEncoder):
    """Batching shifts the bf16 reduction order, so compare by cosine.

    Measured: repeated calls are bit-identical, but a 3-word batch versus three
    single-word calls agree to cosine 0.99998 (float32: 0.9999999). Real queries
    are batch-1, so a query cache stays consistent.
    """
    texts = ["a fire truck", "a blue van", "a person walking"]
    together = encoder.encode_texts(texts)
    separately = np.concatenate([encoder.encode_texts([text]) for text in texts], axis=0)
    cosines = np.sum(together * separately, axis=1)
    assert cosines.min() > 0.9999, f"batching changed the vectors: {cosines.tolist()}"


def test_text_encoding_is_deterministic(encoder: AlignedEncoder):
    assert np.array_equal(encoder.encode_texts(["a cat"]), encoder.encode_texts(["a cat"]))


def test_single_string_and_list_agree(encoder: AlignedEncoder):
    """A bare string is one text, not a sequence of characters."""
    assert np.allclose(encoder.encode_texts("a cat"), encoder.encode_texts(["a cat"]), atol=1e-6)


def test_empty_texts_are_rejected(encoder: AlignedEncoder):
    with pytest.raises(ValueError):
        encoder.encode_texts([])
    with pytest.raises(ValueError):
        encoder.encode_texts(["a cat", "   "])


def test_text_and_image_share_one_space(encoder: AlignedEncoder):
    """Solid red, solid blue and a checkerboard, each with its own label.

    A CLIP text tower from some other model would have the same dimension and
    would fail this: the diagonal would not be the row maximum.
    """
    red, blue, white, checker, gradient, noise = distinct_frames()
    images = encoder.encode_images([red, blue, checker])
    labels = ["a red image", "a blue image", "a black and white checkered pattern"]
    scores = encoder.encode_texts(labels) @ images.T

    assert scores.shape == (3, 3)
    assert list(np.argmax(scores, axis=0)) == [0, 1, 2], (
        "image-text ranking is not aligned:\n" + str(np.round(scores, 4))
    )
    for i in range(3):
        assert scores[i, i] > 0.2, f"{labels[i]!r} is not clearly closer to its own frame"


def test_negative_control_on_two_different_images(encoder: AlignedEncoder):
    """Two images must each prefer their own label over the other's."""
    red, blue = distinct_frames()[:2]
    images = encoder.encode_images([red, blue])
    texts = encoder.encode_texts(["a red image", "a blue image"])
    scores = texts @ images.T
    assert scores[0, 0] > scores[0, 1]
    assert scores[1, 1] > scores[1, 0]


@pytest.mark.skipif(not CAT_PHOTO.is_file(), reason=f"{CAT_PHOTO} is absent")
def test_real_photo_prefers_its_own_subject(encoder: AlignedEncoder):
    """A real photo, if the repo has one. Ranked against plausible distractors."""
    image = encoder.encode_images([CAT_PHOTO])[0]
    labels = ["a photo of a cat", "a kitten", "a dog", "a car", "a fire truck", "a building"]
    scores = encoder.encode_texts(labels) @ image
    order = list(np.argsort(-scores))
    assert order[0] in (0, 1), f"top label was {labels[order[0]]!r}: {dict(zip(labels, np.round(scores, 4)))}"
    assert scores[0] > scores[labels.index("a car")]
    assert scores[0] > scores[labels.index("a fire truck")]


def test_compressed_cosine_scale_is_documented(encoder: AlignedEncoder):
    """Cosine is not a probability, and the spec's threshold section depends on it.

    Even a correct match lands well below 1.0, so any `min_similarity` default has
    to be calibrated (phase E) rather than assumed.
    """
    image = encoder.encode_images([camera_frame(3)])[0]
    unrelated = encoder.encode_texts(["a fire truck"])[0]
    assert float(image @ unrelated) < 0.5
