"""Image preprocessing for RADIO.

This mirrors `examples/common/resize_transform.py` and
`examples/common/__init__.py::get_standard_transform` from the pinned RADIO
source in `third_party/RADIO`. It is reimplemented here so the retrieval
pipeline does not depend on the examples package, and
`tests/test_preprocess_matches_official.py` asserts it matches the official
transform element for element.

Vectorisation contract (spec 2.2): the pipeline hands the model a float32 tensor
in `[0, 1]`. Normalisation to the OPENAI_CLIP mean/std is done by the model's own
`input_conditioner`, which is where the official TorchHub path does it. Nothing
here normalises, and nothing double-normalises.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

BICUBIC = Image.BICUBIC


def _round_to_multiple(value: float, multiple: int) -> int:
    """Mirror `ResizeTransform._get_nearest`, including Python's round-half-to-even."""
    return int(round(value / multiple) * multiple)


def resize_size(
    height: int,
    width: int,
    resolution: Sequence[int],
    resize_multiple: int,
    max_dim: bool = False,
) -> tuple[int, int]:
    """Target (height, width) after the official resize step.

    One element in `resolution` resizes the shortest side (or the longest side
    when `max_dim`) and keeps the aspect ratio. Two elements resize so the image
    covers `resolution`, to be followed by the pad/crop pair.
    """
    if len(resolution) == 1:
        if max_dim:
            factor = resolution[0] / max(height, width)
        else:
            factor = resolution[0] / min(height, width)
        size = (height * factor, width * factor)
    elif len(resolution) == 2:
        in_aspect = height / width
        out_aspect = resolution[0] / resolution[1]
        if (in_aspect > out_aspect and not max_dim) or (in_aspect < out_aspect and max_dim):
            scale = resolution[1] / width
        else:
            scale = resolution[0] / height
        size = (height * scale, width * scale)
    else:
        raise ValueError(f"resolution must have 1 or 2 elements, got {list(resolution)!r}")

    return tuple(_round_to_multiple(dim, resize_multiple) for dim in size)  # type: ignore[return-value]


def pad_to_square(tensor: torch.Tensor, pad_mean: float = 0.0) -> torch.Tensor:
    """Mirror `PadToSquare`: pad bottom/right so the last two dims are square."""
    height, width = tensor.shape[-2:]
    longest = max(height, width)
    pad_h = max(0, longest - height)
    pad_w = max(0, longest - width)
    top = pad_h // 2
    left = pad_w // 2
    return torch.nn.functional.pad(
        tensor, (left, pad_w - left, top, pad_h - top), mode="constant", value=pad_mean
    )


def center_crop(tensor: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
    """Mirror `torchvision.transforms.v2.functional.center_crop` for a CHW tensor.

    torchvision names its padding `[left, top, right, bottom]` but hands it to
    `torch.nn.functional.pad`, which wants `(left, right, top, bottom)`; the
    order below is already the latter.
    """
    height, width = tensor.shape[-2:]
    crop_h, crop_w = output_size
    if crop_w > width or crop_h > height:
        left = (crop_w - width) // 2 if crop_w > width else 0
        top = (crop_h - height) // 2 if crop_h > height else 0
        right = (crop_w - width + 1) // 2 if crop_w > width else 0
        bottom = (crop_h - height + 1) // 2 if crop_h > height else 0
        tensor = torch.nn.functional.pad(tensor, (left, right, top, bottom), mode="constant", value=0.0)
        height, width = tensor.shape[-2:]
        if crop_w == width and crop_h == height:
            return tensor
    top = int(round((height - crop_h) / 2.0))
    left = int(round((width - crop_w) / 2.0))
    return tensor[..., top : top + crop_h, left : left + crop_w]


def to_rgb_pil(image: Image.Image | np.ndarray | str | Path, channel_order: str = "rgb") -> Image.Image:
    """Normalise any accepted input into an RGB `PIL.Image`.

    `channel_order` only applies to 3-channel numpy arrays. Bayer/grey demosaic
    and BGR/Bayer conversion belong to the capture adapter (spec section 4); this
    function only handles what the encoder itself accepts.
    """
    if isinstance(image, (str, Path)):
        with Image.open(image) as handle:
            return handle.convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        array = image
        if array.dtype != np.uint8:
            raise TypeError(f"expected a uint8 array, got {array.dtype}")
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"expected an HxW or HxWx3 array, got shape {array.shape}")
        if channel_order == "bgr":
            array = array[:, :, ::-1]
        elif channel_order != "rgb":
            raise ValueError(f"channel_order must be 'rgb' or 'bgr', got {channel_order!r}")
        return Image.fromarray(np.ascontiguousarray(array), mode="RGB")
    raise TypeError(f"unsupported image type: {type(image).__name__}")


def prepare_image(
    image: Image.Image | np.ndarray | str | Path,
    resolution: Sequence[int],
    resize_multiple: int,
    max_dim: bool = False,
    channel_order: str = "rgb",
) -> torch.Tensor:
    """Run the official transform chain and return float32 CHW in `[0, 1]`."""
    pil = to_rgb_pil(image, channel_order=channel_order)
    size = resize_size(pil.height, pil.width, resolution, resize_multiple, max_dim=max_dim)

    if min(size) <= 0:
        raise ValueError(
            f"resize to {size} from {pil.size} with resolution={list(resolution)} "
            f"and resize_multiple={resize_multiple} produced a degenerate size"
        )

    pil = pil.resize((size[1], size[0]), resample=BICUBIC)
    tensor = torch.from_numpy(np.asarray(pil, dtype=np.uint8).copy()).permute(2, 0, 1).float() / 255.0

    if len(resolution) == 2:
        tensor = pad_to_square(tensor, pad_mean=0.0)
        tensor = center_crop(tensor, (resolution[0], resolution[1]))

    return tensor.contiguous()


def prepare_batch(
    images: Sequence[Image.Image | np.ndarray | str | Path],
    resolution: Sequence[int],
    resize_multiple: int,
    max_dim: bool = False,
    channel_order: str = "rgb",
) -> list[torch.Tensor]:
    """Prepare a batch as a list of CHW tensors.

    RADIO accepts variable-size inputs, but a batch is a single tensor, so images
    that resize to different shapes must be encoded separately. Callers group by
    the returned shapes.
    """
    if not images:
        raise ValueError("no images given")
    return [
        prepare_image(image, resolution, resize_multiple, max_dim=max_dim, channel_order=channel_order)
        for image in images
    ]
