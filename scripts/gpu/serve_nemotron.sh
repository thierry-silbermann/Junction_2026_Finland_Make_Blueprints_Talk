#!/usr/bin/env bash
# Runs ON THE GPU POD (copied there by runpod_up.sh). Installs vLLM once, then starts
# Nemotron 3 Nano Omni (FP8) bound to 127.0.0.1:8000, reachable only through the SSH tunnel.
set -euo pipefail

MODEL="${MODEL:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8}"
VLLM_VERSION="${VLLM_VERSION:-0.20.0}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
WORK=/workspace
export HF_HOME="$WORK/hf"

if [ -f "$WORK/.hf_token" ]; then
  HF_TOKEN="$(cat "$WORK/.hf_token")"
  export HF_TOKEN
fi

if curl -sf http://127.0.0.1:8000/v1/models >/dev/null; then
  echo "vLLM is already serving on the pod."
  exit 0
fi

if [ ! -x "$WORK/vllm-env/bin/vllm" ]; then
  echo "Installing vLLM $VLLM_VERSION (first start only, a few minutes)..."
  python3 -m pip install -q uv
  python3 -m uv venv "$WORK/vllm-env" --python 3.12
  VIRTUAL_ENV="$WORK/vllm-env" python3 -m uv pip install -q "vllm==$VLLM_VERSION" --torch-backend=auto
fi

echo "Starting vLLM with $MODEL (the first start downloads ~33 GB of weights)..."
# The model card's reasoning parser keeps any thinking text out of the answer content;
# requests disable thinking anyway (chat_template_kwargs.enable_thinking=false).
nohup "$WORK/vllm-env/bin/vllm" serve "$MODEL" \
  --served-model-name nemotron-omni \
  --host 127.0.0.1 --port 8000 \
  --trust-remote-code \
  --max-model-len 16384 \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --limit-mm-per-prompt '{"image": 1}' \
  --gpu-memory-utilization 0.92 \
  --enable-prefix-caching \
  --reasoning-parser nemotron_v3 \
  > "$WORK/vllm.log" 2>&1 &
echo "vLLM started in the background; log: $WORK/vllm.log"
