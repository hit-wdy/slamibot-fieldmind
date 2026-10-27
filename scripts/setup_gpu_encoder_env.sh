#!/usr/bin/env bash
# One-time setup of the CPython 3.8 CUDA environment the encoder sidecar needs.
#
# JetPack 5 ships CUDA 11.4 and NVIDIA publishes CUDA torch wheels for it only
# for CPython 3.8. The rest of FieldMind needs 3.10+, hence two environments.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${FIELDMIND_GPU_VENV:-$HOME/wdy/.venvs/fm-gpu}"
TORCH_WHEEL="${FIELDMIND_TORCH_WHEEL:-torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl}"
TORCH_URL="${FIELDMIND_TORCH_URL:-https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/$TORCH_WHEEL}"
CACHE="${FIELDMIND_WHEEL_CACHE:-/tmp}"

usage() {
  cat <<'EOF'
Usage: scripts/setup_gpu_encoder_env.sh [--install-cuda-runtime]

Creates the CPython 3.8 venv, installs the JetPack CUDA torch build and the
Python dependencies the encoder needs, then prints the verification command.

--install-cuda-runtime  also installs cuda-cudart/libcublas/libcudnn/nvtx and
                        libopenblas from NVIDIA's r35.6 apt repo (needs sudo).
                        Without these, `import torch` fails with
                        "libcublas.so.* not found" or "libopenblas.so.0".
EOF
}

install_runtime() {
  echo "== installing CUDA 11.4 runtime libraries (sudo)" >&2
  sudo tee /etc/apt/sources.list.d/nvidia-l4t-apt-source.list >/dev/null <<'LIST'
deb https://repo.download.nvidia.com/jetson/common r35.6 main
LIST
  sudo apt-get update
  sudo apt-get install -y \
    cuda-cudart-11-4 cuda-nvrtc-11-4 cuda-nvtx-11-4 \
    libcublas-11-4 libcufft-11-4 libcurand-11-4 libcusparse-11-4 libcusolver-11-4 \
    libcudnn8 libopenblas-dev libomp-dev
}

main() {
  if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    return 0
  fi
  if [[ "${1:-}" == "--install-cuda-runtime" ]]; then
    install_runtime
  fi

  if ! command -v python3.8 >/dev/null; then
    echo "python3.8 is required (JetPack 5 ships it as the system Python)" >&2
    exit 1
  fi

  echo "== creating $VENV (Python 3.8)" >&2
  mkdir -p "$(dirname "$VENV")"
  python3.8 -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip setuptools wheel

  local wheel="$CACHE/$TORCH_WHEEL"
  if [[ ! -f "$wheel" ]]; then
    echo "== downloading $TORCH_WHEEL (163 MB)" >&2
    curl -fSL --retry 3 -o "$wheel" "$TORCH_URL"
  fi
  echo "== installing torch (CUDA 11.4, CPython 3.8)" >&2
  "$VENV/bin/pip" install "$wheel"

  # torchvision 0.16.2 is the last cp38 aarch64 wheel; timm imports
  # torchvision.ops.misc at load time. Install it without dependencies so pip
  # cannot replace the CUDA torch with the CPU one from PyPI.
  echo "== installing torchvision 0.16.2 (no deps, keeps CUDA torch)" >&2
  "$VENV/bin/pip" install --no-deps "torchvision==0.16.2"
  echo "== installing encoder dependencies" >&2
  "$VENV/bin/pip" install "numpy<2" "open_clip_torch==2.32.0" timm einops "huggingface_hub<0.35" pillow PyYAML

  # Re-assert the CUDA build: a dependency resolution that pulled `torch` from
  # PyPI would have silently replaced it with the CPU wheel.
  echo "== verifying the GPU torch survived dependency installation" >&2
  LD_LIBRARY_PATH="/usr/local/cuda-11.4/lib64:${LD_LIBRARY_PATH:-}" "$VENV/bin/python" - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit(
        f"torch {torch.__version__} has no CUDA; reinstall the JetPack wheel "
        "(a PyPI dependency replaced it)"
    )
print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
PY

  echo >&2
  echo "ready. Start it with: bash $ROOT/scripts/start_encoder_sidecar.sh" >&2
  echo "then point the backend at it: --encoder-url http://127.0.0.1:8791" >&2
  if [[ ! -e /usr/local/cuda-11.4 ]]; then
    echo "WARNING: /usr/local/cuda-11.4 is missing; run with --install-cuda-runtime" >&2
  fi
}

main "$@"
