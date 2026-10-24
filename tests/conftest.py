"""Shared fixtures and deterministic synthetic frames for the encoder tests.

No test asset is downloaded: every frame is generated from a fixed seed so the
suite runs offline and the numbers are reproducible.
"""

from __future__ import annotations

import numpy as np
import pytest

from slamibot_fieldmind import AlignedEncoder, Config

FRAME_SIZE = (576, 1024)  # OAK CAM_A is 16:9


def camera_frame(seed: int, size: tuple[int, int] = FRAME_SIZE) -> np.ndarray:
    """A smooth, colourful frame roughly like a scene: patterns vary with seed."""
    height, width = size
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    base = np.stack(
        [
            (127 + 100 * np.sin(x / 40.0 + seed)).astype(np.int16),
            (127 + 100 * np.cos(y / 30.0 - seed)).astype(np.int16),
            ((x * y / max(height * width, 1)) * 255).astype(np.int16),
        ],
        axis=-1,
    )
    noise = rng.integers(-12, 13, size=base.shape).astype(np.int16)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def distinct_frames(size: tuple[int, int] = FRAME_SIZE) -> list[np.ndarray]:
    """Frames that share no structure, for checking the space is not collapsed.

    Every CLIP space compresses the similarity of unrelated inputs; these are the
    cases where a working encoder still has to separate them.
    """
    height, width = size
    rng = np.random.default_rng(1)
    red = np.zeros((height, width, 3), dtype=np.uint8)
    red[:, :, 0] = 220
    blue = np.zeros((height, width, 3), dtype=np.uint8)
    blue[:, :, 2] = 220
    white = np.full((height, width, 3), 245, dtype=np.uint8)
    checker = np.repeat((((np.indices((height, width)).sum(0) // 40) % 2) * 255)[:, :, None], 3, axis=2)
    y, x = np.mgrid[0:height, 0:width]
    gradient = np.stack([x * 255 // width, y * 255 // height, np.zeros_like(x)], axis=-1).astype(np.uint8)
    noise = rng.integers(0, 256, (height, width, 3)).astype(np.uint8)
    return [red, blue, white, checker.astype(np.uint8), gradient, noise]


@pytest.fixture(scope="session")
def encoder() -> AlignedEncoder:
    config = Config.default()
    if not config.model.checkpoint_path.is_file():
        pytest.skip(
            f"checkpoint {config.model.checkpoint_path} is absent; "
            "run python scripts/fetch_assets.py first"
        )
    return AlignedEncoder(config)
