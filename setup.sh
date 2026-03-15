#!/bin/bash
set -e

# ── System dependencies ───────────────────────────────────────────────────────
sudo apt update && sudo apt install -y ffmpeg git-lfs

# ── uv + Python deps ──────────────────────────────────────────────────────────
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.bashrc"
uv sync

# ── Git config ────────────────────────────────────────────────────────────────
git lfs install
git config --global pull.rebase true
git config --global pull.ff only

read -p "Git name:" git_name
git config --global user.name "$git_name"

read -p "Git email: " git_email
git config --global user.email "$git_email"

git config core.hooksPath .githooks

# ── Secrets ───────────────────────────────────────────────────────────────────
read -p "HuggingFace token: " hf_token
echo "export HF_TOKEN=$hf_token" >> "$HOME/.bashrc"

read -p "WandB API key: " wandb_key
echo "export WANDB_API_KEY=$wandb_key" >> "$HOME/.bashrc"

# ── Claude Code ───────────────────────────────────────────────────────────────
curl -fsSL https://claude.ai/install.sh | bash

# ── Pull LFS data ─────────────────────────────────────────────────────────────
git lfs pull

# ── Reassemble sharded files ──────────────────────────────────────────────────
find . -name "*.part*" -not -path "./.git/*" | \
    sed 's/\.part[a-z]*$//' | sort -u | while read base; do
    if [ -f "${base}" ]; then
        echo "Skipping $base — already exists"
        continue
    fi
    echo "Reassembling: $base"
    cat "${base}".part* > "${base}"
done

source "$HOME/.bashrc"
echo "Setup complete!"

