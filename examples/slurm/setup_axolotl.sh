user=$(whoami)

# install claude code
sudo apt install nodejs npm gh
sudo npm install -g @anthropic-ai/claude-code

# create dirs in vast storage
mkdir /workspace-vast/$user/.cache/uv /workspace-vast/$user/git /workspace-vast/$user/exp /workspace-vast/$user/venvs

# install venv that works across nodes
cd /workspace-vast/$user/venvs
export UV_PYTHON_INSTALL_DIR="/workspace-vast/$user/.uv/python"
export UV_CACHE_DIR="/workspace-vast/$user/.cache/uv"
uv python install 3.11
uv venv venv_axolotl
source venv_axolotl/bin/activate

# setup axolotl
uv pip install torch
uv pip install -U packaging==23.2 setuptools==75.8.0 wheel ninja
uv pip install --no-build-isolation 'axolotl[flash-attn,deepspeed]'

# manually create this:
# echo HF_TOKEN=$HF_TOKEN /workspace-vast/$user/.env