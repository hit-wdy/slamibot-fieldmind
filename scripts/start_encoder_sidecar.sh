#!/usr/bin/env bash
# Start the GPU encoder sidecar for the CPython 3.8 JetPack 5 CUDA torch build.
#
# The sidecar owns the model; the CPython 3.11 backend talks to it over loopback.
# See docs/orin_nx_gpu_encoder.md for why two interpreters are required.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${FIELDMIND_GPU_VENV:-$HOME/wdy/.venvs/fm-gpu}"
CONFIG="${FIELDMIND_GPU_CONFIG:-configs/jetson-gpu.yaml}"
HOST="${FIELDMIND_ENCODER_HOST:-127.0.0.1}"
PORT="${FIELDMIND_ENCODER_PORT:-8791}"
CUDA_LIB="${FIELDMIND_CUDA_LIB:-/usr/local/cuda-11.4/lib64}"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "no interpreter at $VENV; see docs/orin_nx_gpu_encoder.md for the setup" >&2
  exit 1
fi

# NVIDIA's JetPack wheels link against the CUDA runtime in /usr/local/cuda-11.4,
# which is not on the default loader path of an L4T image built without the
# toolkit metapackage.
export LD_LIBRARY_PATH="$CUDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# Weights are already in models/; never reach for the network at boot.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY=1

cd "$ROOT"
echo "starting encoder sidecar: $VENV/bin/python -m slamibot_fieldmind.encoder_service" >&2
echo "  config $CONFIG on http://$HOST:$PORT" >&2
exec "$VENV/bin/python" -m slamibot_fieldmind.encoder_service \
  --model-config "$CONFIG" \
  --host "$HOST" \
  --port "$PORT" \
  "$@"
