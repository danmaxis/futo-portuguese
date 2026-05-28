#!/usr/bin/env bash
# Run the futo-train container on Unraid (host: Zordon @ 192.168.50.24).
# Two modes: simple `docker run` (this script) or via Unraid web UI template (see ../notes/unraid_container_template.md).
#
# This script SSH's to Unraid as root and starts the container.
# Idempotent: stops and removes any existing futo-train container before starting fresh.

set -euo pipefail

UNRAID="${UNRAID_HOST:-root@192.168.50.24}"
GPU_UUID="${GPU_UUID:-GPU-05d0b600-1afe-3701-392d-066e1d784e32}"  # discovered from `ssh root@unraid 'nvidia-smi -L'`
CONTAINER="${CONTAINER:-futo-train}"
HOST_PORT="${HOST_PORT:-2222}"
APPDATA="${APPDATA:-/mnt/user/appdata/futo-train}"

ssh "${UNRAID}" "
  set -e
  if docker ps -a --format '{{.Names}}' | grep -q '^${CONTAINER}\$'; then
    echo 'Removing existing ${CONTAINER}'
    docker rm -f ${CONTAINER}
  fi
  echo 'Starting ${CONTAINER}'
  docker run -d \\
    --name ${CONTAINER} \\
    --restart unless-stopped \\
    --runtime=nvidia \\
    --gpus 'device=${GPU_UUID}' \\
    -p ${HOST_PORT}:22 \\
    -v ${APPDATA}/workspace:/workspace \\
    futo-train:latest
  sleep 3
  docker ps --filter name=${CONTAINER} --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
"

echo
echo 'Smoke test (from this VM):'
ssh -o StrictHostKeyChecking=accept-new -p "${HOST_PORT}" trainer@192.168.50.24 \
  'echo "=== inside container ==="; hostname; nvidia-smi -L; python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_available())"'
