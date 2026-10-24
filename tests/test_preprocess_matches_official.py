"""Our preprocessing must match the official RADIO transform exactly.

The official chain lives in `examples/common/resize_transform.py`. Its package
`__init__` imports a `common` module that is not importable outside the examples
tree, so the file is loaded directly by path.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch
from PIL import Image

from slamibot_fieldmind.paths import RADIO_SOURCE
from slamibot_fieldmind.preprocess import prepare_image, resize_size

pytest.importorskip("torchvision")

RESIZE_TRANSFORM = RADIO_SOURCE / "examples" / "common" / "resize_transform.py"

IMAGE_SIZES = [
    (576, 1024),  # OAK CAM_A 16:9
    (1080, 1920),
    (480, 640),
    (700, 700),
    (1024, 576),  # portrait-ish input
    (333, 777),  # sizes that do not divide evenly
]

RESOLUTIONS = [[768, 768], [512, 512], [432], [512], [384, 640]]


def _load_official():
    if not RESIZE_TRANSFORM.is_file():
        pytest.skip(f"vendored RADIO source is missing {RESIZE_TRANSFORM}")
    spec = importlib.util.spec_from_file_location("official_resize_transform", RESIZE_TRANSFORM)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _image(size: tuple[int, int], seed: int = 0) -> Image.Image:
    height, width = size
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(array, mode="RGB")


@pytest.mark.parametrize("resolution", RESOLUTIONS)
@pytest.mark.parametrize("size", IMAGE_SIZES)
@pytest.mark.parametrize("max_dim", [False, True])
def test_prepare_image_matches_official(size, resolution, max_dim):
    official = _load_official()
    multiple = 16
    image = _image(size)

    expected = official.get_standard_transform(
        list(resolution), multiple, preprocessor=None, max_dim=max_dim, pad_mean=None
    )(image)
    actual = prepare_image(image, list(resolution), multiple, max_dim=max_dim)

    assert actual.shape == expected.shape, f"{size} -> {actual.shape} vs official {expected.shape}"
    assert torch.allclose(actual, expected, atol=1e-6), (
        f"{size} resolution={resolution} max_dim={max_dim}: max abs diff "
        f"{(actual - expected).abs().max().item()}"
    )


@pytest.mark.parametrize("resolution", RESOLUTIONS)
@pytest.mark.parametrize("size", IMAGE_SIZES)
@pytest.mark.parametrize("max_dim", [False, True])
def test_resize_size_matches_official(size, resolution, max_dim):
    official = _load_official()
    multiple = 16
    transform = official.ResizeTransform(list(resolution), multiple, max_dim=max_dim)
    params = transform.make_params([_image(size)])
    assert params["size"] == resize_size(size[0], size[1], list(resolution), multiple, max_dim=max_dim)


def test_range_is_unit_interval_without_normalisation():
    """The encoder feeds [0, 1]; the model's own conditioner does mean/std."""
    tensor = prepare_image(_image((576, 1024)), [768, 768], 16)
    assert tensor.dtype == torch.float32
    assert float(tensor.min()) >= 0.0 and float(tensor.max()) <= 1.0
    assert abs(float(tensor.mean()) - 0.5) < 0.5
