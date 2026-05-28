#!/usr/bin/env bash
# publish_setup.sh — prepare the FUTO pt-BR repo for public release on GitHub.
#
# What it does (in order):
#   1. Print a preview of what will be added / excluded.
#   2. SCP v8.2 GGUFs from danmaxis@192.168.50.10 into models/.
#   3. Write .gitignore + LICENSE.
#   4. Create notes/samples/ with sanitized synth-typo excerpts.
#   5. Stash loose root files (vanilla, v2) into notes/legacy_*.md.
#   6. git init + initial commit (only the published file set).
#   7. Compute SHA256 for the four release GGUFs.
#   8. Print the gh release create commands ready to copy-paste.
#
# Dry-run by default. Pass --apply to actually do the destructive steps.
#
# Idempotent: re-running is safe; SCP / writes are skipped if the destination
# already matches.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

APPLY=0
if [[ "${1:-}" == "--apply" ]]; then APPLY=1; fi

say() { printf '\n\033[1;36m[setup]\033[0m %s\n' "$*"; }
do_or_say() {
    if (( APPLY )); then "$@"; else printf '  \033[2m(dry-run) %s\033[0m\n' "$*"; fi
}

# ---------------------------------------------------------------------------
# 1. Preview
# ---------------------------------------------------------------------------
say "Mode: $([ $APPLY -eq 1 ] && echo APPLY || echo DRY-RUN)"
say "Repo root: $REPO_ROOT"

PUBLISHED=(
    README.md README.pt-BR.md GUIDE.md GUIDE.pt-BR.md LICENSE .gitignore
    scripts/ notes/ tokenizer_v2/spm_pt_br_v2.model
)
EXCLUDED=(
    checkpoints/ finetune_big_v3/ pretrain_big/ corpora/ gguf/ GGUF/
    models/ futo-src/ llama.cpp/ reference_model/
    notes/typo_log.jsonl notes/real_typos_pool.json notes/real_typos_eval.json
    notes/v8_1/real_typos_pool.json notes/v8_1/real_typos_eval.json
    notes/v8_1/real_typo_stats.json
    env/ vanilla v2
)

say "Will be published:"
for p in "${PUBLISHED[@]}"; do printf '  + %s\n' "$p"; done
say "Will be excluded (.gitignore):"
for p in "${EXCLUDED[@]}"; do printf '  - %s\n' "$p"; done

# ---------------------------------------------------------------------------
# 2. SCP v8.2 GGUFs
# ---------------------------------------------------------------------------
say "Pull v8.2 GGUFs from danmaxis@192.168.50.10"
mkdir -p models

declare -A V82_FILES=(
    ["futo_pt_br_v8_2.gguf"]="/home/danmaxis/futo-train/models/futo_pt_br_v8_2_full.gguf"
    ["futo_pt_br_v8_2_q6k.gguf"]="/home/danmaxis/futo-train/models/futo_pt_br_v8_2.q6k.patched2"
)

for local_name in "${!V82_FILES[@]}"; do
    remote_path="${V82_FILES[$local_name]}"
    local_path="models/$local_name"
    if [[ -f "$local_path" ]]; then
        printf '  \033[2m✓ %s already present (skip)\033[0m\n' "$local_path"
    else
        printf '  -> scp danmaxis@192.168.50.10:%s %s\n' "$remote_path" "$local_path"
        do_or_say scp "danmaxis@192.168.50.10:$remote_path" "$local_path"
    fi
done

# ---------------------------------------------------------------------------
# 3. Write .gitignore + LICENSE
# ---------------------------------------------------------------------------
say "Write .gitignore"
if (( APPLY )); then
cat > .gitignore <<'GITIGNORE'
# --- Heavy training artifacts (reproducible from scripts/) ---
checkpoints/
finetune_big_v3/
finetune_big_v8_1/
finetune_big_v8_2/
pretrain_big/
pretrain/
finetune/
finetune_v3/
corpora/

# --- Historical GGUF experiments (only release assets ship) ---
gguf/
GGUF/
models/
# (models/ is ignored so the GGUFs stay on disk for upload to GitHub Releases
#  without being tracked in git.)

# --- Upstream clones (referenced via URL in README) ---
futo-src/
llama.cpp/
reference_model/

# --- Personal real-typo data (user's own typing — never publish) ---
notes/typo_log.jsonl
notes/real_typos_pool.json
notes/real_typos_eval.json
notes/v8_1/real_typos_pool.json
notes/v8_1/real_typos_eval.json
notes/v8_1/real_typo_stats.json

# --- Machine-specific config / scratch ---
env/
docker/local/
*.log
*.status
*.bootstrap.log
*.done

# --- Python / build ---
__pycache__/
*.pyc
*.egg-info/
.venv/
.python-version

# --- Editor ---
.vscode/
.idea/
*.swp
.DS_Store
GITIGNORE
fi

say "Write LICENSE (MIT)"
if (( APPLY )); then
cat > LICENSE <<'LICENSE'
MIT License

Copyright (c) 2026 the FUTO Portuguese contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
LICENSE
fi

# ---------------------------------------------------------------------------
# 4. notes/samples/ — sanitized synth excerpts
# ---------------------------------------------------------------------------
say "Build notes/samples/"
mkdir -p notes/samples

if (( APPLY )); then
    python3 - <<'PY'
import json, random
random.seed(7)

def take_per_category(path, n_per_cat):
    data = json.load(open(path))
    by_cat = {}
    for d in data:
        by_cat.setdefault(d.get("category","other"), []).append(d)
    out = []
    for cat, items in by_cat.items():
        out.extend(random.sample(items, min(n_per_cat, len(items))))
    return out

# Hand-rolled synth — 8 per category
sample1 = take_per_category("notes/synth_typos.json", 8)
json.dump(sample1, open("notes/samples/synth_typos_sample.json","w"),
          ensure_ascii=False, indent=2)
print(f"  wrote synth_typos_sample.json ({len(sample1)} entries)")

# v8.1 Claude-generated — 5 per category if exists
import os
for src, dst, n in [
    ("notes/v8_1/synth_claude_general.json",  "notes/samples/synth_claude_general_sample.json", 5),
    ("notes/v8_1/synth_claude_weakspot.json", "notes/samples/synth_claude_weakspot_sample.json", 5),
]:
    if os.path.exists(src):
        s = take_per_category(src, n)
        json.dump(s, open(dst,"w"), ensure_ascii=False, indent=2)
        print(f"  wrote {dst} ({len(s)} entries)")

# README inside the dir to explain
open("notes/samples/README.md","w").write("""# Synthetic typo samples

Small excerpts of the synthetic typo corpora used to fine-tune the model.
The full pools (200K hand-rolled + Claude-generated batches) are reproducible
via `scripts/04a_build_wordfreq.py`, `scripts/lib_typo_synthesis.py`, and
`scripts/run_claude_synth_v8_1.sh`. We publish only samples here to (a) show
the data shape and (b) keep the repo small.

The corresponding *real* typo pool is NOT published — it is the maintainer's
own typed text and is gitignored.
""")
PY
fi

# ---------------------------------------------------------------------------
# 5. Stash loose root notes
# ---------------------------------------------------------------------------
say "Stash loose root files into notes/legacy_*.md"
for f in vanilla v2; do
    if [[ -f "$f" ]]; then
        dst="notes/legacy_${f}_notes.md"
        if [[ -f "$dst" ]]; then
            printf '  \033[2m✓ %s already exists (skip)\033[0m\n' "$dst"
        else
            printf '  -> mv %s %s\n' "$f" "$dst"
            do_or_say mv "$f" "$dst"
        fi
    fi
done

# ---------------------------------------------------------------------------
# 6. git init + initial commit
# ---------------------------------------------------------------------------
say "Initialize git repo"
if [[ -d .git ]]; then
    printf '  \033[2m✓ .git already exists (skip init)\033[0m\n'
else
    do_or_say git init -b main
fi

say "Stage published files (respecting .gitignore)"
do_or_say git add README.md README.pt-BR.md GUIDE.md GUIDE.pt-BR.md LICENSE .gitignore \
                  scripts/ notes/ tokenizer_v2/spm_pt_br_v2.model

if (( APPLY )); then
    if git diff --cached --quiet; then
        printf '  \033[2m✓ nothing to commit (already up to date)\033[0m\n'
    else
        say "Create initial commit"
        git commit -m "Initial public release: pt-BR FUTO Keyboard LM journey

Bilingual (EN + PT-BR) writeup of training a Brazilian Portuguese
language model for FUTO Keyboard, plus the full reproducible pipeline.
See README.md and GUIDE.md for the story and the technical reference.
The trained model GGUFs are attached to GitHub Releases (v8.2 latest)."
    fi
fi

# ---------------------------------------------------------------------------
# 7. SHA256 for release assets
# ---------------------------------------------------------------------------
say "Compute SHA256 for release GGUFs"
for f in \
    models/futo_pt_br_v8.gguf \
    models/futo_pt_br_v8_q6k.gguf \
    models/futo_pt_br_v8_2.gguf \
    models/futo_pt_br_v8_2_q6k.gguf
do
    if [[ -f "$f" ]]; then
        if (( APPLY )); then
            sha256sum "$f"
        else
            printf '  \033[2m(dry-run) sha256sum %s\033[0m\n' "$f"
        fi
    else
        printf '  \033[2m✗ missing: %s (run step 2 first)\033[0m\n' "$f"
    fi
done

# ---------------------------------------------------------------------------
# 8. Print next steps
# ---------------------------------------------------------------------------
cat <<'NEXT'

[setup] Done. Suggested next steps:

  1. Create the GitHub repo (e.g. on github.com or via `gh repo create`):

       gh repo create futo-portuguese --public --source=. --remote=origin --push

  2. Cut the v8 (legacy) release:

       gh release create v8 \
         --title "v8 — first shipping pt-BR model" \
         --notes "First shipping version. 36% top-1 / 56% top-5 on the 50-pair
                  real-typo holdout; 88.2% top-5 on accent_only. Smoke-tested on
                  Android (FUTO Keyboard 0.1.27). See README §The journey → v8." \
         models/futo_pt_br_v8.gguf \
         models/futo_pt_br_v8_q6k.gguf

  3. Cut the v8.2 (latest) release:

       gh release create v8.2 --latest \
         --title "v8.2 — current pt-BR model" \
         --notes "Continue-pretrained on conversational corpus + PLW_C=0.05.
                  60% top-1 / 72% top-5 on the v8 holdout (vs v8's 36/56).
                  +8.6pp top-5 on the harder v8.1 holdout. NWP top-3 (masked)
                  is 4.66% — slight regression vs v8's 6.13%, please smoke-test
                  on-device. See README §The journey → v8.2." \
         models/futo_pt_br_v8_2.gguf \
         models/futo_pt_br_v8_2_q6k.gguf

  4. Verify the rendered README on GitHub. Click through every link.

NEXT

if (( APPLY == 0 )); then
    cat <<'TIP'

This was a DRY RUN. Re-run with --apply to actually do the writes, SCP,
git init, and commit:

    bash scripts/publish_setup.sh --apply

TIP
fi
