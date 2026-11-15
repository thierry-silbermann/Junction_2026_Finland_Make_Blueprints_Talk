#!/usr/bin/env bash
# Runs ON YOUR LAPTOP. Starts Nemotron on a Runpod pod and holds an SSH tunnel to it.
#
#   scripts/gpu/runpod_up.sh <pod-ip> <ssh-port> [path-to-ssh-key]
#
# The IP and port come from the pod's Connect tab, "SSH over exposed TCP"
# (ssh root@<pod-ip> -p <ssh-port> ...). While this script runs, the model answers at
# http://localhost:${LOCAL_PORT:-8002}/v1, which the `nemotron_gpu` alias and the `gpu` profile use.
# Ctrl-C closes the tunnel only; stop the pod in the Runpod console to stop paying.
set -euo pipefail

if [ $# -lt 2 ]; then
  sed -n '2,10p' "$0"
  exit 1
fi
HOST="$1"
PORT="$2"
KEY="${3:-$HOME/.ssh/id_ed25519}"
LOCAL_PORT="${LOCAL_PORT:-8002}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OPTS=(-i "$KEY" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30)
SSH=(ssh "${OPTS[@]}" -p "$PORT" "root@$HOST")

"${SSH[@]}" 'cat > /workspace/serve_nemotron.sh' < "$HERE/serve_nemotron.sh"
if [ -n "${HF_TOKEN:-}" ]; then
  # Sent over stdin so the token never appears in a remote command line.
  printf '%s' "$HF_TOKEN" | "${SSH[@]}" 'umask 077; cat > /workspace/.hf_token'
fi
"${SSH[@]}" "MODEL='${MODEL:-}' MAX_NUM_SEQS='${MAX_NUM_SEQS:-64}' bash /workspace/serve_nemotron.sh" \
  || { echo "Setup failed; see the output above." >&2; exit 1; }

echo "Waiting for the model to load (first start: download + load, typically 5-15 minutes)..."
started=$(date +%s)
until "${SSH[@]}" "curl -sf http://127.0.0.1:8000/v1/models >/dev/null"; do
  if ! "${SSH[@]}" "pgrep -f 'vllm serve' >/dev/null"; then
    echo "vLLM exited. Last log lines:" >&2
    "${SSH[@]}" "tail -n 40 /workspace/vllm.log" >&2
    exit 1
  fi
  printf '  %ss: %s\n' "$(( $(date +%s) - started ))" "$("${SSH[@]}" "tail -n 1 /workspace/vllm.log | cut -c1-120")"
  sleep 20
done

echo "Model is up. Tunnel: http://localhost:$LOCAL_PORT/v1  (Ctrl-C to close; stop the pod in Runpod when done)"
echo "In another terminal:  drawingqa doctor --probe   or   drawingqa evaluate --profile gpu ..."
exec ssh "${OPTS[@]}" -p "$PORT" -N -L "$LOCAL_PORT:127.0.0.1:8000" "root@$HOST"
