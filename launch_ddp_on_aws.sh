#!/usr/bin/env bash
#
# launch_ddp_on_aws.sh
# Discover EC2 instances by tag, assign NODE_RANK / MASTER_ADDR,
# and launch torchrun on each node via SSH.
#
# Usage:
#   export CLUSTER_TAG_VALUE=mycluster1
#   ./launch_ddp_on_aws.sh train.py --arg1 ... --argN ...
#
# Requirements:
#   - All nodes have the same tag key/value (default key: TrainingCluster).
#   - AWS CLI and jq installed on this node.
#   - Instance profile (IAM role) allows ec2:DescribeInstances.
#   - SSH key can be used to login as ec2-user to each node.
#

set -euo pipefail

########################
# Configurable parameters
########################

# Tag key / value to identify the training cluster
CLUSTER_TAG_KEY="${CLUSTER_TAG_KEY:-TrainingCluster}"
CLUSTER_TAG_VALUE="${CLUSTER_TAG_VALUE:-}"

# SSH settings
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/id_rsa}"
SSH_USER="${SSH_USER:-ec2-user}"

# torchrun / DDP settings
MASTER_PORT="${MASTER_PORT:-29500}"
GPUS_PER_NODE="${GPUS_PER_NODE:-}"   # if empty, will auto-detect via nvidia-smi

########################
# Basic checks
########################

if [[ -z "$CLUSTER_TAG_VALUE" ]]; then
  echo "ERROR: CLUSTER_TAG_VALUE is not set. Example:" >&2
  echo "  export CLUSTER_TAG_VALUE=mycluster1" >&2
  exit 1
fi

if ! command -v aws &>/dev/null; then
  echo "ERROR: aws CLI not found. Install awscli first." >&2
  exit 1
fi

if ! command -v jq &>/dev/null; then
  echo "ERROR: jq not found. Install jq first." >&2
  exit 1
fi

if [[ "$#" -lt 1 ]]; then
  echo "Usage: CLUSTER_TAG_VALUE=<value> $0 <train_script.py> [args...]" >&2
  exit 1
fi

TRAIN_SCRIPT="$1"
shift
TRAIN_ARGS="$*"

########################
# Detect region from EC2 metadata
########################

echo "[launch_ddp_on_aws] Detecting region from instance metadata..."
REGION="$(curl -s http://169.254.169.254/latest/dynamic/instance-identity/document | jq -r .region)"

echo "[launch_ddp_on_aws] Using region: ${REGION}"

########################
# Discover instances by tag
########################

echo "[launch_ddp_on_aws] Discovering instances with tag ${CLUSTER_TAG_KEY}=${CLUSTER_TAG_VALUE}..."

HOSTS_JSON="$(
  aws ec2 describe-instances \
    --region "$REGION" \
    --filters "Name=tag:${CLUSTER_TAG_KEY},Values=${CLUSTER_TAG_VALUE}" "Name=instance-state-name,Values=running" \
    --query 'Reservations[].Instances[].PrivateIpAddress' \
    --output json
)"

# Convert to sorted Bash array
mapfile -t HOSTS < <(echo "$HOSTS_JSON" | jq -r '.[]' | sort)

WORLD_SIZE="${#HOSTS[@]}"

if [[ "$WORLD_SIZE" -eq 0 ]]; then
  echo "ERROR: No running instances found for tag ${CLUSTER_TAG_KEY}=${CLUSTER_TAG_VALUE}" >&2
  exit 1
fi

echo "[launch_ddp_on_aws] Found ${WORLD_SIZE} hosts:"
printf '  %s\n' "${HOSTS[@]}"

MASTER_ADDR="${HOSTS[0]}"
echo "[launch_ddp_on_aws] MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

########################
# Determine GPUs per node
########################

if [[ -z "${GPUS_PER_NODE}" ]]; then
  echo "[launch_ddp_on_aws] GPUS_PER_NODE is not set; auto-detecting on this node via nvidia-smi..."
  if command -v nvidia-smi &>/dev/null; then
    GPUS_PER_NODE="$(nvidia-smi -L | wc -l)"
  else
    echo "ERROR: nvidia-smi not found and GPUS_PER_NODE not set." >&2
    exit 1
  fi
fi

echo "[launch_ddp_on_aws] GPUS_PER_NODE=${GPUS_PER_NODE}"

########################
# Launch on each host via SSH
########################

echo "[launch_ddp_on_aws] Launching training on all hosts..."

for i in "${!HOSTS[@]}"; do
  host="${HOSTS[$i]}"
  rank="$i"

  echo "[launch_ddp_on_aws] Launching NODE_RANK=${rank} on host ${host}..."

  ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no "${SSH_USER}@${host}" \
    "export NNODES=${WORLD_SIZE} NODE_RANK=${rank} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} GPUS_PER_NODE=${GPUS_PER_NODE}; \
     torchrun_ddp.sh ${TRAIN_SCRIPT} ${TRAIN_ARGS}" &
done

echo "[launch_ddp_on_aws] Waiting for all ranks to finish..."
wait
echo "[launch_ddp_on_aws] All ranks finished."
