#!/bin/bash
set -e

# ── System dependencies ───────────────────────────────────────────────────────
sudo apt update && sudo apt install -y ffmpeg

# ── uv + Python deps ──────────────────────────────────────────────────────────
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# ── Git config ────────────────────────────────────────────────────────────────
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

# ── Download weights and cache from HuggingFace ───────────────────────────────
export HF_TOKEN="$hf_token"
uv run python scripts/download_artifacts.py

echo "Setup complete, run 'source .bashrc'!"
