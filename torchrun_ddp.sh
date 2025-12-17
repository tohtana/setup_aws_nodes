########################
# torchrun helper script for multi-node DDP
########################

log "Installing torchrun_ddp.sh helper script..."

USER_HOME="/home/${TARGET_USER}"
USER_BIN="${USER_HOME}/bin"
HELPER_SCRIPT="${USER_BIN}/torchrun_ddp.sh"

sudo -u "$TARGET_USER" mkdir -p "$USER_BIN"

cat >"$HELPER_SCRIPT" <<'EOF'
#!/usr/bin/env bash
#
# torchrun_ddp.sh
# Small convenience wrapper for multi-node PyTorch DDP.
#
# Usage (on each node):
#   export NNODES=4
#   export NODE_RANK=0   # 0..NNODES-1
#   export MASTER_ADDR=10.0.0.1   # private IP of rank 0 node
#   # Optional:
#   # export MASTER_PORT=29500
#   # export GPUS_PER_NODE=8
#
#   torchrun_ddp.sh train.py --arg1 ... --argN ...
#
set -euo pipefail

# Required env vars
: "${NNODES:?Set NNODES (total number of nodes)}"
: "${NODE_RANK:?Set NODE_RANK (this node's rank, 0..NNODES-1)}"
: "${MASTER_ADDR:?Set MASTER_ADDR (rank 0 node's private IP or hostname)}"

MASTER_PORT="${MASTER_PORT:-29500}"

# Detect GPU count if not provided
if [ -z "${GPUS_PER_NODE:-}" ]; then
  if command -v nvidia-smi &>/dev/null; then
    GPUS_PER_NODE="$(nvidia-smi -L | wc -l)"
  else
    echo "nvidia-smi not found and GPUS_PER_NODE is not set." >&2
    exit 1
  fi
fi

if [ "$#" -lt 1 ]; then
  echo "Usage: torchrun_ddp.sh <train_script.py> [args...]" >&2
  exit 1
fi

SCRIPT="$1"
shift

echo "[torchrun_ddp] NNODES=${NNODES} NODE_RANK=${NODE_RANK} GPUS_PER_NODE=${GPUS_PER_NODE} MASTER_ADDR=${MASTER_ADDR}:${MASTER_PORT}"
echo "[torchrun_ddp] Running: torchrun --nnodes=${NNODES} --nproc_per_node=${GPUS_PER_NODE} --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} ${SCRIPT} $*"

exec torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${SCRIPT}" "$@"
EOF

chown "$TARGET_USER":"$TARGET_USER" "$HELPER_SCRIPT"
chmod +x "$HELPER_SCRIPT"

# Make sure ~/bin is in PATH for interactive shells
if ! grep -q 'export PATH="$HOME/bin:$PATH"' "$BASHRC" 2>/dev/null; then
  echo 'export PATH="$HOME/bin:$PATH"' >> "$BASHRC"
fi
