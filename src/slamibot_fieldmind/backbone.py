"""Loading of the pinned RADIO backbone plus its image/text adaptor.

Everything here comes from the official `NVlabs/RADIO` code vendored in
`third_party/RADIO` at a pinned commit. The checkpoint is loaded through the
official `hubconf.radio_model` entry point; we only point torch's hub cache at
the project directory so the verified local file is used instead of a fresh
download, and we keep the CLIP text tower off the GPU unless asked for it.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .config import ModelConfig
from .paths import ASSETS_MANIFEST, HF_HOME, RADIO_SOURCE, TORCH_HUB_DIR

HUBCONF_NAME = "radio_hubconf"
# The Hugging Face asset open_clip needs before it can build the `clip` adaptor.
HF_TEXT_ENCODER_ASSET = "dfn5b_clip_vit_h14_378"

# The dtype the checkpoint stores is not the dtype we run in; this maps the
# config values onto torch dtypes. `auto` keeps the checkpoint's own dtype.
DTYPE_BY_NAME = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def as_torch_dtype(value: str | torch.dtype) -> torch.dtype:
    """Accept 'float32', 'torch.float32' or an already-resolved torch.dtype."""
    if isinstance(value, torch.dtype):
        return value
    return DTYPE_BY_NAME[str(value).rsplit(".", 1)[-1]]


@dataclass
class LoadReport:
    """Facts collected while loading, for the phase A verification report."""

    version: str = ""
    adaptor: str = ""
    checkpoint: str = ""
    checkpoint_sha256: str = ""
    checkpoint_bytes: int = 0
    source_commit: str = ""
    checkpoint_dtype: str = ""
    model_dtype: str = ""
    device: str = ""
    text_encoder_device: str = ""
    text_encoder_dtype: str = ""
    parameter_count: int = 0
    adaptor_names: list[str] = field(default_factory=list)
    adaptor_classes: dict[str, str] = field(default_factory=dict)
    checkpoint_teachers: list[dict] = field(default_factory=list)
    text_encoder: dict = field(default_factory=dict)
    missing_keys: list[str] = field(default_factory=list)
    unexpected_keys: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def ensure_radio_importable() -> None:
    """Put the vendored RADIO root on `sys.path` so `import radio` resolves to it."""
    _shim_torch_distributed()
    if not (RADIO_SOURCE / "radio" / "__init__.py").is_file():
        raise FileNotFoundError(
            f"the vendored RADIO source is missing at {RADIO_SOURCE}; "
            "see README.md for the pinned NVlabs/RADIO checkout commands"
        )
    root = str(RADIO_SOURCE)
    if root not in sys.path:
        sys.path.insert(0, root)
    module = sys.modules.get("radio")
    if module is not None and not str(getattr(module, "__file__", "")).startswith(root):
        raise RuntimeError(
            f"a different 'radio' package was already imported from {getattr(module, '__file__', '?')}; "
            "import slamibot_fieldmind before anything else that ships a top-level 'radio' module"
        )


def _shim_torch_distributed() -> None:
    """Fill in the `torch.distributed` surface RADIO imports at module load.

    NVIDIA's JetPack 5 aarch64 wheels are built with `USE_DISTRIBUTED=0`, so
    `torch.distributed` exists but has no `is_initialized`/`get_rank` and no
    `torch.distributed.nn.functional`. RADIO only uses those for multi-process
    training (`rank_gate`, an all-reduce on the token count), never in the
    single-process inference path we run, so no-op stand-ins are enough.
    """
    distributed = getattr(torch, "distributed", None)
    if distributed is None:
        return
    if hasattr(torch._C, "_distributed_c10d"):
        return

    stubs = {
        "is_available": lambda: False,
        "is_initialized": lambda: False,
        "get_rank": lambda group=None: 0,
        "get_world_size": lambda group=None: 1,
        "get_backend": lambda group=None: "undefined",
        "destroy_process_group": lambda group=None: None,
    }
    for name, stub in stubs.items():
        if not hasattr(distributed, name):
            setattr(distributed, name, stub)
    if not hasattr(distributed, "ReduceOp"):

        class ReduceOp:  # noqa: D401 - mirrors torch.distributed.ReduceOp
            SUM = 0
            PRODUCT = 1
            MIN = 2
            MAX = 3
            BAND = 4
            BOR = 5
            BXOR = 6
            PREMUL_SUM = 7
            AVG = 8

        distributed.ReduceOp = ReduceOp

    nn_package = sys.modules.get("torch.distributed.nn")
    if nn_package is None:
        nn_package = importlib.util.module_from_spec(
            importlib.machinery.ModuleSpec("torch.distributed.nn", loader=None)
        )
        nn_package.__path__ = []
        sys.modules["torch.distributed.nn"] = nn_package
        distributed.nn = nn_package
    functional = sys.modules.get("torch.distributed.nn.functional")
    if functional is None:
        # RADIO imports `all_reduce` at module level for the multi-process path
        # only; in a single process it must be the identity, not a new group.
        functional = importlib.util.module_from_spec(
            importlib.machinery.ModuleSpec(
                "torch.distributed.nn.functional", loader=None
            )
        )
        functional.all_reduce = lambda tensor, *args, **kwargs: tensor
        functional.all_gather = lambda tensor, *args, **kwargs: (tensor,)
        sys.modules["torch.distributed.nn.functional"] = functional
        nn_package.functional = functional


def load_hubconf():
    """Import the vendored `hubconf.py` under a private module name."""
    cached = sys.modules.get(HUBCONF_NAME)
    if cached is not None:
        return cached
    ensure_radio_importable()
    spec = importlib.util.spec_from_file_location(HUBCONF_NAME, RADIO_SOURCE / "hubconf.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {RADIO_SOURCE / 'hubconf.py'}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[HUBCONF_NAME] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint(path: Path, expected_sha256: str, expected_bytes: int | None = None) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {path}\n"
            "run: python scripts/fetch_assets.py  (downloads and SHA-256 verifies the pinned assets)"
        )
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"{path.name} is {path.stat().st_size} bytes, expected {expected_bytes}; "
            "re-run python scripts/fetch_assets.py"
        )
    if not expected_sha256:
        raise RuntimeError("no pinned SHA-256 for the checkpoint; refusing to load an unverified file")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"{path.name} SHA-256 mismatch, refusing to load it:\n"
            f"  expected {expected_sha256}\n  actual   {actual}"
        )


def resolve_dtype(name: str, fallback: torch.dtype) -> torch.dtype:
    return fallback if name == "auto" else DTYPE_BY_NAME[name]


def configure_caches() -> None:
    """Keep torch Hub and Hugging Face downloads inside the project directory.

    When the pinned CLIP text tower is already in the project cache, also switch
    huggingface_hub to offline mode. That is what actually pins the revision:
    `open_clip` asks for `revision='main'`, so without this a moved upstream
    branch would be followed silently, and every load would make a Hub
    round-trip.
    """
    TORCH_HUB_DIR.mkdir(parents=True, exist_ok=True)
    hub_cache = HF_HOME / "hub"
    hub_cache.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(TORCH_HUB_DIR))

    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    import huggingface_hub.constants as hf_constants

    hf_constants.HF_HUB_CACHE = str(hub_cache)

    if cached_hf_asset_path(HF_TEXT_ENCODER_ASSET) is not None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        hf_constants.HF_HUB_OFFLINE = True


def cached_hf_asset_path(asset_name: str) -> Path | None:
    """Locate a `kind: hf_hub` asset in the project cache without going online.

    Only the size is checked here; `scripts/fetch_assets.py` is what verifies the
    SHA-256. This guard exists to decide offline mode, not to authorise a load.
    """
    manifest = json.loads(ASSETS_MANIFEST.read_text(encoding="utf-8"))
    asset = next((a for a in manifest["assets"] if a["name"] == asset_name), None)
    if asset is None or asset.get("kind") != "hf_hub":
        return None
    repo_dir = HF_HOME / "hub" / f"models--{asset['repo_id'].replace('/', '--')}"
    for candidate in repo_dir.glob(f"snapshots/*/{asset['filename']}"):
        if candidate.stat().st_size == asset["size"]:
            return candidate
    return None


def _detach_text_encoders(model: Any) -> dict[str, Any]:
    """Remove the CLIP text towers from the module tree so `.to()` skips them."""
    detached: dict[str, Any] = {}
    for name, adaptor in (getattr(model, "adaptors", None) or {}).items():
        encoder = getattr(adaptor, "oc_model", None)
        if isinstance(encoder, torch.nn.Module):
            detached[name] = encoder
            adaptor.oc_model = None
    return detached


def _attach_text_encoders(model: Any, detached: dict[str, Any]) -> None:
    for name, encoder in detached.items():
        model.adaptors[name].oc_model = encoder


def _set_conditioner_dtype(model: Any, dtype: torch.dtype) -> None:
    """Make the input conditioner emit `dtype`.

    `InputConditioner.forward` casts through its `dtype` attribute, which
    hubconf seeds from the checkpoint. Casting the module alone would leave that
    attribute stale and silently upcast the input back to float32.
    """
    conditioner = getattr(model, "input_conditioner", None)
    if conditioner is not None and hasattr(conditioner, "dtype"):
        conditioner.dtype = dtype


def resolve_device(name: str) -> torch.device:
    """Resolve `auto` against the runtime instead of failing on a CPU-only host.

    A device that was flashed without the vendor CUDA runtime can still run the
    index and MCP surface, just slowly. Silently pretending CUDA works would be
    worse; the load report records the resolved device.
    """
    if str(name) == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"model.device is {name!r} but torch.cuda.is_available() is False. "
            "Install a CUDA build of torch for this L4T release, or set "
            "model.device: auto to run on CPU."
        )
    return device


def load_radio_model(cfg: ModelConfig, verify: bool = True) -> tuple[Any, LoadReport]:
    """Load RADIO with its adaptor and place it on the configured devices."""
    report = LoadReport(
        version=cfg.version,
        adaptor=cfg.adaptor,
        checkpoint=str(cfg.checkpoint_path),
        checkpoint_sha256=cfg.checkpoint_sha256,
        source_commit=cfg.source_commit,
    )

    if verify:
        verify_checkpoint(cfg.checkpoint_path, cfg.checkpoint_sha256)
    elif not cfg.checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {cfg.checkpoint_path}")
    report.checkpoint_bytes = cfg.checkpoint_path.stat().st_size

    configure_caches()
    hubconf = load_hubconf()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if cfg.text_encoder_load == "text_only":
            # Build only the paired text tower. The adaptor otherwise constructs
            # the full CLIP model (including the 2.5 GB visual tower) and deletes
            # the visual half right away, which dominates the cold-start peak on
            # shared-memory Orin hardware.
            from .text_tower import text_only_open_clip

            # Build in the checkpoint's float32 and cast once below, so a CPU
            # fallback does not first build bfloat16 and then convert.
            with text_only_open_clip():
                model, checkpoint = hubconf.radio_model(
                    version=cfg.version,
                    progress=True,
                    adaptor_names=cfg.adaptor,
                    return_checkpoint=True,
                )
        else:
            model, checkpoint = hubconf.radio_model(
                version=cfg.version,
                progress=True,
                adaptor_names=cfg.adaptor,
                return_checkpoint=True,
            )

    report.warnings = [f"{w.category.__name__}: {w.message}" for w in caught]
    for warning in report.warnings:
        if "Missing keys in state dict" in warning:
            report.missing_keys.append(warning)
        elif "Unexpected keys in state dict" in warning:
            report.unexpected_keys.append(warning)

    report.checkpoint_teachers = [
        {
            "name": teacher.get("name"),
            "type": teacher.get("type"),
            "model": teacher.get("model"),
            "pretrained": teacher.get("pretrained"),
            "use_summary": teacher.get("use_summary", True),
            "input_size": teacher.get("input_size"),
        }
        for teacher in checkpoint["args"].teachers
    ]
    for teacher in report.checkpoint_teachers:
        if teacher["name"] == cfg.adaptor:
            report.text_encoder = {
                "type": teacher["type"],
                "model": teacher["model"],
                "pretrained": teacher["pretrained"],
            }
            break

    checkpoint_dtype = getattr(checkpoint["args"], "dtype", torch.float32)
    del checkpoint

    adaptors = getattr(model, "adaptors", None) or {}
    if cfg.adaptor not in adaptors:
        raise RuntimeError(
            f"the loaded model has no adaptor named {cfg.adaptor!r}; loaded adaptors: {sorted(adaptors)}. "
            "Without the image/text adaptor the backbone summary is not in the CLIP text space."
        )
    report.adaptor_names = sorted(adaptors)
    report.adaptor_classes = {name: type(a).__name__ for name, a in adaptors.items()}

    model.eval()
    report.checkpoint_dtype = str(checkpoint_dtype)

    device = resolve_device(cfg.device)
    text_device = resolve_device(cfg.text_encoder_device)
    # Half precision is a GPU optimisation. On CPU it hits unvectorised slow
    # paths (measured ~70 s/frame for bfloat16 versus ~5 s for float32 on Orin
    # NX), so an `auto`/half request falls back to float32 there and says so.
    dtype = resolve_dtype(cfg.dtype, checkpoint_dtype)
    if device.type == "cpu" and dtype in (torch.bfloat16, torch.float16):
        report.warnings.append(
            f"model.dtype {dtype} is not accelerated on CPU; using float32"
        )
        dtype = torch.float32
    report.model_dtype = str(dtype)

    text_dtype = DTYPE_BY_NAME[cfg.text_encoder_dtype]
    if text_device.type == "cpu" and text_dtype in (torch.bfloat16, torch.float16):
        report.warnings.append(
            f"model.text_encoder_dtype {text_dtype} is not accelerated on CPU; using float32"
        )
        text_dtype = torch.float32

    # Take the CLIP text tower out of the module tree so `.to()` skips it, then
    # place it exactly where the config says. Leaving it in would drag 1.4 GB of
    # text weights onto the image device even when they are configured for CPU.
    detached = _detach_text_encoders(model)
    model.to(device=device, dtype=dtype)
    _set_conditioner_dtype(model, dtype)
    for encoder in detached.values():
        encoder.to(device=text_device, dtype=text_dtype).eval()
    _attach_text_encoders(model, detached)

    report.device = str(device)
    report.text_encoder_device = str(text_device) if detached else str(device)
    report.text_encoder_dtype = str(text_dtype)
    report.parameter_count = sum(p.numel() for p in model.parameters())
    model.requires_grad_(False)
    return model, report


def model_facts(model: Any) -> dict[str, Any]:
    """Read the facts the spec requires us to confirm rather than assume."""
    facts: dict[str, Any] = {}
    for attr in ("patch_size", "min_resolution_step", "max_resolution", "summary_dim", "embed_dim"):
        facts[attr] = getattr(model, attr, None)
    preferred = getattr(model, "preferred_resolution", None)
    if preferred is not None:
        # RESOURCE_MAP entries use either a Resolution NamedTuple or a plain tuple.
        height, width = (preferred.height, preferred.width) if hasattr(preferred, "height") else preferred
        facts["preferred_resolution"] = [height, width]
    adaptors: dict[str, Any] = {}
    for name, adaptor in (getattr(model, "adaptors", None) or {}).items():
        entry: dict[str, Any] = {
            "class": type(adaptor).__name__,
            "head_idx": getattr(adaptor, "head_idx", None),
            "has_encode_text": callable(getattr(adaptor, "encode_text", None)),
            "has_tokenizer": getattr(adaptor, "tokenizer", None) is not None,
        }
        summary_mlp = getattr(adaptor, "head_mlp", None)
        if summary_mlp is not None:
            entry["summary_mlp_output_dim"] = _mlp_output_dim(summary_mlp)
        adaptors[name] = entry
    facts["adaptors"] = adaptors
    return facts


def _mlp_output_dim(mlp: Any) -> int | None:
    for module in reversed(list(mlp.modules())):
        out_features = getattr(module, "out_features", None)
        if out_features is not None:
            return int(out_features)
    return None
