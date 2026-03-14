curl -LsSf https://astral.sh/uv/install.sh | sh
sudo apt update && sudo apt install -y ffmpeg git-lfs
uv sync
git lfs install
git config pull.rebase true
git config pull.ff only
git config user.name "Marek324"
git lfs pull
curl -fsSL https://claude.ai/install.sh | bash
curl -fsSL https://claude.ai/install.sh | bash
echo "Don\'t forget to add HF token, git user.email, wandb API key, and run source ~/.bashrc"
