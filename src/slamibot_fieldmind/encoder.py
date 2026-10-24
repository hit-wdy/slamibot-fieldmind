"""The image/text encoder contract from section 2.3 of the spec.

Both directions go through the same image-text adaptor (here `clip`).
`encode_images` takes the adaptor's global `summary` — not the backbone's own
summary and not an average of backbone spatial tokens, which are different vector
spaces. `encode_texts` uses that adaptor's own tokenizer and text encoder, so the
two outputs are comparable by cosine.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .backbone import LoadReport, as_torch_dtype, load_radio_model, model_facts
from .config import Config, ModelConfig
from .preprocess import prepare_batch

ImageLike = Any

# Bumped whenever the numbers produced by this module could change.
CODE_VERSION = "image-text-1"


class AlignedEncoder:
    def __init__(self, config: Config | None = None, verify_checkpoint: bool = True) -> None:
        self.config = config or Config.default()
        self.model_config: ModelConfig = self.config.model
        self.model, self.load_report = load_radio_model(self.model_config, verify=verify_checkpoint)
        self.facts = model_facts(self.model)

        self._resize_multiple = (
            self.model_config.resize_multiple
            if self.model_config.resize_multiple is not None
            else int(self.facts["min_resolution_step"])
        )
        self._validate_resolution()

        self._embedding_dim: int | None = None
        self._fingerprint: str | None = None

    # -- contract ---------------------------------------------------------------

    def encode_images(self, images: Sequence[ImageLike] | ImageLike, channel_order: str = "rgb") -> np.ndarray:
        """Return float32 `[B, D]`, L2 normalised, one row per input image."""
        single = _is_single_image(images)
        batch = [images] if single else list(images)  # type: ignore[arg-type]
        embeddings = self._encode_batch(batch, channel_order=channel_order)
        return embeddings

    def encode_texts(self, texts: Sequence[str] | str) -> np.ndarray:
        """Return float32 `[K, D]`, L2 normalised, in the same space as `encode_images`.

        Uses the adaptor's own text encoder and tokenizer, which is the only
        pairing whose vectors are comparable with the image summary. A CLIP text
        tower from anywhere else would have the same dimension and a different
        space (spec 2.1).
        """
        if isinstance(texts, str):
            texts = [texts]
        batch = [str(text) for text in texts]
        if not batch:
            raise ValueError("no texts given")
        if any(not text.strip() for text in batch):
            raise ValueError("texts must not be empty")

        adaptor = self.model.adaptors[self.model_config.adaptor]
        tokenizer = getattr(adaptor, "tokenizer", None)
        if tokenizer is None or not callable(getattr(adaptor, "encode_text", None)):
            raise RuntimeError(
                f"adaptor {self.model_config.adaptor!r} ({type(adaptor).__name__}) has no text encoder; "
                "its image embeddings cannot be searched with text"
            )

        device = torch.device(self.load_report.text_encoder_device)
        tokens = tokenizer(batch)
        if hasattr(tokens, "to"):
            tokens = tokens.to(device)

        with torch.inference_mode():
            embeddings = adaptor.encode_text(tokens)
        # Normalise in float32 rather than through the adaptor's own
        # `normalize=True`: in bfloat16 that rounds to ~1e-3 off unit norm, and
        # the image side already normalises after upcasting.
        normalized = F.normalize(embeddings.float(), dim=-1)
        result = normalized.cpu().numpy().astype(np.float32)
        _validate_embeddings(result, self.embedding_dim)
        return result

    def fingerprint(self) -> str:
        """Stable hash of everything that determines the vector space."""
        if self._fingerprint is None:
            self._fingerprint = compute_fingerprint(
                self.model_config,
                self.load_report,
                embedding_dim=self.embedding_dim,
                resize_multiple=self._resize_multiple,
            )
        return self._fingerprint

    def fingerprint_short(self) -> str:
        return self.fingerprint()[:16]

    # -- derived facts ----------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        if self._embedding_dim is None:
            reported = self.facts["adaptors"].get(self.model_config.adaptor, {}).get("summary_mlp_output_dim")
            if not reported:
                raise RuntimeError(f"could not read the embedding dimension of adaptor {self.model_config.adaptor!r}")
            self._embedding_dim = int(reported)
        return self._embedding_dim

    @property
    def resize_multiple(self) -> int:
        return self._resize_multiple

    def set_input_resolution(self, resolution: Sequence[int]) -> None:
        """Switch preprocessing resolution without reloading the model.

        Only the preprocessing changes, so the weights are untouched; the
        fingerprint is invalidated because the vectors move.
        """
        previous = list(self.model_config.input_resolution)
        self.model_config.input_resolution = [int(v) for v in resolution]
        try:
            self.model_config.__post_init__()
            self._validate_resolution()
        except ValueError:
            self.model_config.input_resolution = previous
            raise
        self._fingerprint = None

    @property
    def environment(self) -> dict[str, Any]:
        return {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device": self.load_report.device,
            "device_name": _device_name(self.load_report.device),
            "text_encoder_device": self.load_report.text_encoder_device,
        }

    # -- internals --------------------------------------------------------------

    def _validate_resolution(self) -> None:
        resolution = list(self.model_config.input_resolution)
        multiple = self._resize_multiple
        if len(resolution) == 2:
            bad = [v for v in resolution if v % multiple]
            if bad:
                raise ValueError(
                    f"center-crop input_resolution {resolution} must be a multiple of "
                    f"resize_multiple={multiple} (the model's min_resolution_step)"
                )
        max_resolution = self.facts.get("max_resolution")
        if len(resolution) == 2 and max_resolution and max(v for v in resolution) > max_resolution:
            raise ValueError(
                f"input_resolution {resolution} exceeds the checkpoint's max_resolution={max_resolution}"
            )

    def _encode_batch(self, images: list[ImageLike], channel_order: str) -> np.ndarray:
        tensors = prepare_batch(
            images,
            self.model_config.input_resolution,
            self._resize_multiple,
            max_dim=self.model_config.max_dim,
            channel_order=channel_order,
        )
        self._check_sizes(tensors)
        device = torch.device(self.load_report.device)
        dtype = as_torch_dtype(self.load_report.model_dtype)

        chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for group in _group_by_shape(tensors, int(self.model_config.batch_size)):
                batch = torch.stack(group).to(device=device, dtype=dtype)
                output = self.model(batch)
                summary = output[self.model_config.adaptor].summary
                normalized = F.normalize(summary.float(), dim=-1)
                chunks.append(normalized.cpu().numpy().astype(np.float32))

        embeddings = np.concatenate(chunks, axis=0)
        _validate_embeddings(embeddings, self.embedding_dim)
        return embeddings

    def _check_sizes(self, tensors: list[torch.Tensor]) -> None:
        multiple = self._resize_multiple
        max_resolution = self.facts.get("max_resolution")
        for tensor in tensors:
            height, width = tensor.shape[-2:]
            if height % multiple or width % multiple:
                raise ValueError(
                    f"prepared image {height}x{width} is not a multiple of resize_multiple={multiple}; "
                    "the RADIO forward pass rejects it"
                )
            if max_resolution and max(height, width) > max_resolution:
                raise ValueError(
                    f"prepared image {height}x{width} exceeds max_resolution={max_resolution}; "
                    "lower model.input_resolution or set model.max_dim"
                )


def compute_fingerprint(
    model_config: ModelConfig,
    report: LoadReport,
    embedding_dim: int,
    resize_multiple: int,
) -> str:
    """Hash of the checkpoint identity, the adaptor and the preprocessing.

    Two runs may only share a vector collection when these agree: the same
    numbers in a differently built space are not comparable (spec 2.3).
    """
    payload = {
        "code_version": CODE_VERSION,
        "checkpoint": {
            "version": model_config.version,
            "sha256": model_config.checkpoint_sha256,
            "bytes": report.checkpoint_bytes,
            "source_commit": model_config.source_commit,
            "checkpoint_dtype": report.checkpoint_dtype,
        },
        "run_dtype": report.model_dtype,
        "adaptor": model_config.adaptor,
        "adaptor_class": report.adaptor_classes.get(model_config.adaptor),
        "text_encoder": report.text_encoder,
        "embedding_dim": embedding_dim,
        "preprocessing": {
            "input_resolution": list(model_config.input_resolution),
            "resize_multiple": resize_multiple,
            "max_dim": model_config.max_dim,
            "resize_interpolation": "bicubic",
            "pad_mean": 0.0,
            "value_range": [0.0, 1.0],
            "normalization": "model.input_conditioner (OPENAI_CLIP mean/std)",
            "channel_order": "rgb",
        },
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _is_single_image(images: Any) -> bool:
    """A numpy array is one image (2-D grey or 3-D HWC), never a batch of rows."""
    if isinstance(images, np.ndarray):
        return images.ndim in (2, 3)
    return not isinstance(images, (list, tuple))


def _group_by_shape(tensors: list[torch.Tensor], batch_size: int) -> list[list[torch.Tensor]]:
    """RADIO takes variable-size inputs, but one batch is one tensor."""
    groups: list[list[torch.Tensor]] = []
    for tensor in tensors:
        shape = tuple(tensor.shape)
        for group in groups:
            if tuple(group[0].shape) == shape and len(group) < batch_size:
                group.append(tensor)
                break
        else:
            groups.append([tensor])
    return groups


def _validate_embeddings(embeddings: np.ndarray, dim: int) -> None:
    """Startup contract from spec 2.2: right shape, finite, unit norm."""
    if embeddings.ndim != 2 or embeddings.shape[1] != dim:
        raise RuntimeError(f"encoder produced shape {embeddings.shape}, expected [B, {dim}]")
    if embeddings.dtype != np.float32:
        raise RuntimeError(f"encoder produced {embeddings.dtype}, expected float32")
    if not np.isfinite(embeddings).all():
        raise RuntimeError("encoder produced non-finite values")
    if embeddings.size and float(np.abs(embeddings).max()) == 0.0:
        raise RuntimeError("encoder produced an all-zero embedding")
    norms = np.linalg.norm(embeddings, axis=1)
    if embeddings.shape[0] and float(np.abs(norms - 1.0).max()) > 1e-3:
        raise RuntimeError(f"encoder output is not L2 normalised: norms {norms.tolist()}")


def _device_name(device: str) -> str:
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(device)
    return device.type


def describe(report: LoadReport, encoder: AlignedEncoder) -> str:
    lines = [
        f"version           {report.version}",
        f"adaptor           {report.adaptor} ({report.adaptor_classes.get(report.adaptor)})",
        f"checkpoint        {report.checkpoint}",
        f"checkpoint sha256 {report.checkpoint_sha256}",
        f"source commit     {report.source_commit}",
        f"checkpoint dtype  {report.checkpoint_dtype} -> run dtype {report.model_dtype}",
        f"parameters        {report.parameter_count / 1e6:.1f} M",
        f"device            {report.device} ({_device_name(report.device)})",
        f"text encoder      {report.text_encoder_device}",
        f"embedding dim     {encoder.embedding_dim}",
        f"fingerprint       {encoder.fingerprint()}",
    ]
    return "\n".join(lines)
