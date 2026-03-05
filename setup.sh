curl -LsSf https://astral.sh/uv/install.sh | sh
apt install -y ffmpeg git-lfs
uv sync
git lfs install
git config pull.rebase true
git config pull.ff only
git config user.name "Marek324"
echo 'Don\'t forget to add HF token and user.email'
