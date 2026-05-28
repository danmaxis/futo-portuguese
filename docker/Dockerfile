# FUTO pt_BR training container — runs on Unraid host with NVIDIA Driver plugin
# Build: docker build --build-arg SSH_PUBKEY="$(cat ~/.ssh/id_ed25519.pub)" -t futo-train:latest .
# Run:   --runtime=nvidia --gpus all -p 2222:22 -v /mnt/user/appdata/futo-train:/workspace --restart unless-stopped

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ARG SSH_PUBKEY=""

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
      openssh-server git tmux rsync wget curl vim ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
      transformers datasets sentencepiece accelerate \
      huggingface_hub wandb gguf protobuf langdetect

RUN useradd -m -s /bin/bash trainer && \
    mkdir -p /home/trainer/.ssh /workspace && \
    chown -R trainer:trainer /home/trainer /workspace && \
    chmod 700 /home/trainer/.ssh && \
    ln -sf /opt/conda/bin/python /usr/local/bin/python && \
    ln -sf /opt/conda/bin/python3 /usr/local/bin/python3 && \
    ln -sf /opt/conda/bin/pip /usr/local/bin/pip && \
    ln -sf /opt/conda/bin/hf /usr/local/bin/hf 2>/dev/null || true

RUN if [ -n "$SSH_PUBKEY" ]; then \
      echo "$SSH_PUBKEY" > /home/trainer/.ssh/authorized_keys && \
      chmod 600 /home/trainer/.ssh/authorized_keys && \
      chown trainer:trainer /home/trainer/.ssh/authorized_keys; \
    fi

RUN ssh-keygen -A && \
    mkdir -p /run/sshd && \
    sed -i 's/#PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D", "-e"]
