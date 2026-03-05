curl -LsSf https://astral.sh/uv/install.sh | sh
apt install -y ffmpeg
uv sync
apt intall git-lfs -y
git lfs install
echo "don't forget to add HF_TOKEN and set up git config"
