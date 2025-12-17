#!/usr/bin/env bash
#
# setup_dl_env.sh
#   - For Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.9 (Amazon Linux 2023)
#   - Intended to be run as root (e.g., from EC2 User Data)
#
set -euo pipefail

########################
# Configurable parameters
########################

# Target user (for this DLAMI, usually "ec2-user")
TARGET_USER="${TARGET_USER:-ec2-user}"

# Project root directory
PROJECT_ROOT="${PROJECT_ROOT:-/home/${TARGET_USER}/workspace}"

# Python virtualenv directory
VENV_DIR="${VENV_DIR:-/home/${TARGET_USER}/.venvs/dl-env}"

# Python binary to use
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

# (Optional) Git repository for your training code
REPO_URL="${REPO_URL:-}"

# (Optional) URL to requirements.txt
REQUIREMENTS_URL="${REQUIREMENTS_URL:-}"

########################
# Helpers
########################

log() { echo "[setup_dl_env] $*"; }

########################
# Pre-checks
########################

if [[ "$EUID" -ne 0 ]]; then
  echo "Please run this script as root (via sudo or User Data)." >&2
  exit 1
fi

if ! id "$TARGET_USER" &>/dev/null; then
  echo "User $TARGET_USER does not exist. Check TARGET_USER." >&2
  exit 1
fi

########################
# Detect package manager
########################

PKG_MGR=""
if command -v dnf &>/dev/null; then
  PKG_MGR="dnf"
elif command -v yum &>/dev/null; then
  PKG_MGR="yum"
else
  echo "Neither dnf nor yum is available. Are you on Amazon Linux?" >&2
  exit 1
fi

########################
# Base packages
########################

log "Installing base packages via ${PKG_MGR}..."

$PKG_MGR -y update || true
$PKG_MGR -y install \
  python3 python3-pip \
  python3.11 python3.11-pip \
  git tmux htop
  
# Ensure curl command exists; prefer curl-minimal on AL2023
if ! command -v curl &>/dev/null; then
  log "curl command not found, installing curl-minimal..."
  $PKG_MGR -y install curl-minimal || true
fi

########################
# Install CUDA Toolkit (if not already installed)
########################

log "Checking for CUDA Toolkit (nvcc)..."

if ! command -v nvcc &>/dev/null; then
  log "nvcc not found. Installing CUDA Toolkit via ${PKG_MGR} (cuda-13-0.x86_64)..."

  # Install non-interactively
  if ! $PKG_MGR -y install cuda-13-0.x86_64; then
    log "WARNING: Failed to install cuda-13-0.x86_64. Please check CUDA repo / network."
  fi
else
  log "nvcc already present. Skipping CUDA Toolkit installation."
fi

########################
# Check Python binary
########################

if ! command -v "$PYTHON_BIN" &>/dev/null; then
  echo "Python binary ${PYTHON_BIN} not found." >&2
  exit 1
fi

########################
# Create directories
########################

log "Creating directories: $PROJECT_ROOT, $(dirname "$VENV_DIR")"
sudo -u "$TARGET_USER" mkdir -p "$PROJECT_ROOT"
sudo -u "$TARGET_USER" mkdir -p "$(dirname "$VENV_DIR")"

########################
# Create Python venv
########################

if [[ ! -d "$VENV_DIR" ]]; then
  log "Creating Python venv at: $VENV_DIR"
  sudo -u "$TARGET_USER" "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  log "Using existing venv: $VENV_DIR"
fi

log "Upgrading pip / wheel..."
sudo -u "$TARGET_USER" bash -lc "source '$VENV_DIR/bin/activate' && pip install --upgrade pip wheel"

########################
# Install Python packages from requirements (optional)
########################

if [[ -n "$REQUIREMENTS_URL" ]]; then
  log "Downloading requirements.txt from: $REQUIREMENTS_URL"
  TMP_REQ="/tmp/requirements.txt"
  curl -fsSL "$REQUIREMENTS_URL" -o "$TMP_REQ"
  chown "$TARGET_USER":"$TARGET_USER" "$TMP_REQ"

  log "Installing pip packages from requirements.txt..."
  sudo -u "$TARGET_USER" bash -lc "source '$VENV_DIR/bin/activate' && pip install -r '$TMP_REQ'"

else
  log "REQUIREMENTS_URL is not set; will install default packages later."
fi

########################
# Clone Git repository (optional)
########################

if [[ -n "$REPO_URL" ]]; then
  log "Cloning Git repository: $REPO_URL"
  repo_name="$(basename "$REPO_URL" .git)"
  sudo -u "$TARGET_USER" bash -lc "
    cd '$PROJECT_ROOT'
    if [ ! -d '$repo_name' ]; then
      git clone '$REPO_URL'
    else
      echo 'Repository already exists, skipping clone: $repo_name'
    fi
  "
fi

########################
# Append environment settings to .bashrc
########################

BASHRC="/home/${TARGET_USER}/.bashrc"

log "Adding dl-env auto-activation to .bashrc..."

if ! grep -q "dl-env auto-activate" "$BASHRC" 2>/dev/null; then
  cat <<'EOF' >> "$BASHRC"

# === dl-env auto-activate (added by setup_dl_env.sh) ===
if [ -d "$HOME/.venvs/dl-env" ]; then
  # Activate your DL virtualenv
  source "$HOME/.venvs/dl-env/bin/activate"

  # Use a tmp directory on the same filesystem as $HOME
  mkdir -p "$HOME/tmp"
  export TMPDIR="$HOME/tmp"

  # ---- CUDA related settings ----
  if [ -z "${CUDA_HOME:-}" ]; then
    if command -v nvcc >/dev/null 2>&1; then
      # e.g. /usr/local/cuda-13.0/bin/nvcc -> /usr/local/cuda-13.0
      export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
    elif [ -d /usr/local/cuda ]; then
      export CUDA_HOME="/usr/local/cuda"
    fi
  fi

  if [ -n "${CUDA_HOME:-}" ]; then
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
  fi

  # ---- Common DL / multi-node environment variables ----
  export OMP_NUM_THREADS=8
  export NCCL_DEBUG=INFO
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
fi
# === end dl-env ===
EOF

  chown "$TARGET_USER":"$TARGET_USER" "$BASHRC"
else
  log ".bashrc already contains dl-env settings; skipping append."
fi

########################
# ulimit settings (file descriptors)
########################

log "Relaxing ulimit (nofile) via /etc/security/limits.d/90-nofile.conf..."

cat >/etc/security/limits.d/90-nofile.conf <<EOF
* soft nofile 1048576
* hard nofile 1048576
EOF

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

########################
# Install core Python packages into venv as TARGET_USER
########################

log "Installing core Python packages into venv as ${TARGET_USER}..."

sudo -u "$TARGET_USER" bash -lc "
  set -euo pipefail
  mkdir -p \"\$HOME/tmp\"
  export TMPDIR=\"\$HOME/tmp\"
  source '$VENV_DIR/bin/activate'
  pip install --no-cache-dir torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu130
  pip install --no-cache-dir flash-attn --no-build-isolation
  pip install --no-cache-dir \
    pre-commit \
    clang-format \
    accelerate \
    pytest-xdist \
    pydot \
    nltk \
    torch_tb_profiler \
    datasets \
    lightning \
    wheel \
    transformers \
    wandb \
    'huggingface-hub[cli]' \
    omegaconf \
    hydra-core
"

########################
# Done
########################

log "Setup finished! dl-env will be auto-activated on next login."
