# User-action items (things I cannot do for you)

These are blockers in Phase 0 that require physical/web-UI access I don't have. They unblock Phase 1+.

---

## A. SSH config — add to `~/.ssh/config` on this VM

I haven't touched your `~/.ssh/config` to avoid clobbering existing entries. Append these blocks (edit values in `<>`):

```ssh-config
Host unraid
    HostName 192.168.50.24
    User root
    # Used for managing the Unraid host itself: docker build, container restart, etc.

Host gpu-train
    HostName 192.168.50.24
    Port 2222
    User trainer
    # The training Docker container on Unraid (3090). After Step B below.

Host gpu-5070ti
    HostName <TBD>
    User <your-user>
    # The gaming desktop. After Step C below.
```

Then verify each (after the corresponding setup step):
```bash
ssh unraid 'hostname'
ssh gpu-train 'nvidia-smi -L'
ssh gpu-5070ti 'nvidia-smi -L'
```

---

## B. Unraid host — install NVIDIA driver and create training container

**One-time prereqs (Unraid web UI):**

1. **Apps tab** — install **Community Applications** plugin if not already present.
2. **Apps tab** → search "NVIDIA Driver" → install the plugin (community, by ich777). Reboot Unraid afterwards.
3. After reboot, in **Settings → Nvidia Driver** confirm the 3090 shows up. Note the GPU UUID (looks like `GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`) — you'll paste it into the container config.
4. Smoke test: spin up any test container with `--runtime=nvidia --gpus all` and run `nvidia-smi`. The 3090 should be visible.

**Build the training image:**

5. From this VM, generate an SSH key if you don't have one yet:
   ```bash
   [ -f ~/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -C "futo-train" -N "" -f ~/.ssh/id_ed25519
   ```
6. Copy the Dockerfile and build context to Unraid:
   ```bash
   ssh root@192.168.50.24 'mkdir -p /mnt/user/appdata/futo-train/docker'
   rsync -avz /home/ai-debian/futo-portuguese/docker/Dockerfile root@192.168.50.24:/mnt/user/appdata/futo-train/docker/
   ```
7. SSH into Unraid and build the image, baking in your public key:
   ```bash
   ssh root@192.168.50.24
   cd /mnt/user/appdata/futo-train/docker
   docker build --build-arg SSH_PUBKEY="$(cat ~/.ssh/authorized_keys | head -1)" -t futo-train:latest .
   # OR pass the key from the calling VM via a temp file if root@unraid doesn't have your pubkey yet
   ```
   (If root@unraid doesn't already have your VM's pubkey: `scp ~/.ssh/id_ed25519.pub root@192.168.50.24:/tmp/futo.pub` first, then `--build-arg SSH_PUBKEY="$(cat /tmp/futo.pub)"`.)

**Create the container (Unraid web UI):**

8. **Docker tab → Add Container**:
   - **Name:** `futo-train`
   - **Repository:** `futo-train:latest`
   - **Network Type:** `Bridge`
   - **Ports:** Container `22` → Host `2222`
   - **Path:** Container `/workspace` → Host `/mnt/user/appdata/futo-train`
   - **Extra Parameters:** `--runtime=nvidia --gpus all` (or use the *Nvidia GPU* dropdown the plugin adds, set to the 3090's UUID)
   - **Restart Policy:** `unless-stopped`
9. Apply. Wait for it to start.

**Verify from this VM:**

10. ```bash
    ssh -p 2222 trainer@192.168.50.24 'nvidia-smi -L'   # should print the 3090
    ssh gpu-train 'python -c "import torch; print(torch.cuda.get_device_name(0))"'
    ```

---

## C. Gaming desktop (RTX 5070 Ti)

1. **Find the IP:** on the gaming desktop, run `ip a | grep 192.168.50` (Linux) or `ipconfig` (Windows). Update the `gpu-5070ti` block in `~/.ssh/config` here.
2. **Linux/WSL2 only:** ensure `nvidia-smi` works on the desktop, then:
   ```bash
   sudo apt install -y python3-venv python3-pip openssh-server
   mkdir -p /data/futo-pt-br && cd /data/futo-pt-br
   python3 -m venv env && source env/bin/activate
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   pip install transformers datasets sentencepiece accelerate huggingface_hub wandb
   ```
3. Add this VM's `~/.ssh/id_ed25519.pub` to the desktop user's `~/.ssh/authorized_keys`:
   ```bash
   ssh-copy-id -i ~/.ssh/id_ed25519.pub <user>@<gaming-desktop-ip>
   ```

---

## D. Done autonomously on this VM (Phase 0 complete)

- ✅ Project skeleton at `/home/ai-debian/futo-portuguese/`
- ✅ Local CPU-only venv at `env/` — gguf, huggingface_hub, sentencepiece, protobuf
- ✅ `llama.cpp` cloned (CPU build deferred until Phase 5)
- ✅ Reference English model downloaded — `reference_model/ml4_1_f16_meta_fixed.gguf` (62 MB)
- ✅ `notes/reference_metadata.txt` — full gguf_dump
- ✅ `notes/reference_full_features.txt` — full feature flag string (the wiki was missing two)
- ✅ `notes/reference_slot_map.md` — annotated map of all 300 user-defined symbols
- ✅ `notes/reference_first_64_tokens.txt`, `reference_special_tokens.txt`
- ✅ `reference_model/extracted_spm.model` — embedded SentencePiece, extracted (474 KB)
- ✅ **Critical schema findings** confirmed via reference + Android source — saved to memory:
  - `<XBU>/<XBC>/<XEC>/<XC0>` are name-looked-up; `<CHAR_A>..<CHAR_Z>` MUST be 26 sequential IDs (pointer arithmetic in C++).
  - Feature encoder rows are HARD-CODED at indices 208/209/210 — only relevant if `xc0_swipe_typing_v1` is declared. We're skipping swipe ML in v1, so 208+ is free.
  - Architecture: context_length=2048, rms_norm_eps=1e-6, tie_word_embeddings=False (the wiki had wrong values for some of these).
- ✅ `docker/Dockerfile` — builds the Unraid training container
- ✅ `scripts/01_build_corpus.py` — pt_BR corpus assembly (streaming HF datasets)
- ✅ `scripts/02_train_tokenizer.py` — SentencePiece training with verified 300-slot layout
- ✅ `scripts/02b_smoke_test_tokenizer.py` — **PASSING** locally; confirms slot mechanics work
- ✅ `scripts/03_pretrain.py` — base LM pretrain on the 3090 container
- ✅ `scripts/06_patch_metadata.py` — final GGUF metadata injection

**Smoke test result (just ran on this VM with synthetic pt_BR sample):**
- 300 user-defined symbols at IDs 4..303 ✓
- `<CHAR_A>..<CHAR_Z>` sequential at 182..207 (matches English reference exactly) ✓
- `<0x00>` at id 304 (matches reference) ✓
- XBU autocorrect format encodes/decodes ✓
- Whitespace-as-suffix active ✓

## E. In progress / completed

**Verified working:**
- ✅ Root SSH key on Unraid — `ssh unraid 'hostname'` returns `Zordon`
- ✅ NVIDIA driver plugin on Unraid — `nvidia-smi -L` returns `RTX 3090 GPU-05d0b600-...`
- ✅ Docker 27.5.1 with `nvidia` runtime registered
- ✅ `futo-train:latest` Docker image built (13.8 GB, on docker vdisk)
- ✅ Container running on Unraid, port 2222 → trainer@container, GPU attached
- ✅ Dockerfile + scripts pushed to `/mnt/user/appdata/futo-train/workspace/`
- ✅ SSH config aliases on this VM: `unraid`, `gpu-train`, `gpu-5070ti`
- ✅ **Wikipedia-pt corpus built** — 880K docs, 500M tokens, 2.0 GB across 8 shards at `/workspace/corpora/wiki_pt/`
- 🔄 **Tokenizer training in progress** — BPE merge phase, ~10.6 GiB RAM, silent (no log progress in this phase; expect 30-60 min)
- ✅ **Gaming rig reachable** — `MX-HEADROLLER` Manjaro Linux, RTX 5070 Ti, 31 GiB RAM, driver 590.48
- 🔄 **PyTorch installing on gaming rig** — Python 3.10.13 via pyenv, venv at `/home/danmaxis/futo-pt-br/env`

**No remaining user actions** for the mini run pipeline.

**Next when build finishes:**
1. Run `bash remote/run_container.sh` to start the container with the 3090 attached
2. Smoke test: `ssh gpu-train 'nvidia-smi -L && python -c "import torch; print(torch.cuda.is_available())"'`
3. (Optional) Inside the container: `hf auth login` — only needed if BrWaC/OSCAR datasets gate behind auth
4. Phase 1: kick off `01_build_corpus.py` to assemble the pt_BR corpus on the 3090

**On me (deferred until needed):**
- Phase 4a/b/c fine-tune scripts — easier to write once we have a pretrain checkpoint
- llama.cpp C++ build — needed only at Phase 5.3 (inference verification)
- Phase 5.1 wrapper around `convert_hf_to_gguf.py` — trivial 5-line script, write at the time
