# run on both nodes!

cd ~/git
git clone git@github.com:axolotl-ai-cloud/axolotl.git
cd axolotl && uv venv && source .venv/bin/activate
uv pip install setuptools
uv pip install torch==2.4.1+cu124 --index-url https://download.pytorch.org/whl/cu124
uv pip install --no-build-isolation -e '.[flash-attn]'