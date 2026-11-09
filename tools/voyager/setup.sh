#!/usr/bin/env bash
# One-time setup on a rented GPU (RunPod "PyTorch 2.4.0" template: Python 3.11,
# CUDA 12.4, 80 GB GPU, ~150 GB disk). Installs Voyager, its models, MoGe and gsplat.
#   bash setup.sh
set -euo pipefail
cd /workspace

if [ ! -d HunyuanWorld-Voyager ]; then
  git clone https://github.com/Tencent-Hunyuan/HunyuanWorld-Voyager
fi
cd HunyuanWorld-Voyager

python -m pip install -q -r requirements.txt
python -m pip install -q transformers==4.39.3 xfuser==0.4.2 ninja
python -m pip install -q flash-attn==2.6.3 --no-build-isolation
# Voyager's input tools
python -m pip install -q --no-deps git+https://github.com/microsoft/MoGe.git
python -m pip install -q scipy==1.11.4 \
  git+https://github.com/EasternJournalist/utils3d.git@c5daf6f6c244d251f252102d09e9b7bcef791a38
# Splat training
python -m pip install -q gsplat --index-url https://docs.gsplat.studio/whl/pt24cu124
python -m pip install -q plyfile

# Models (~10–40 min)
huggingface-cli download tencent/HunyuanWorld-Voyager --local-dir ./ckpts

# Our scripts and the boy-free room image (uploaded next to this file)
cp /workspace/make_inputs.py data_engine/
cp /workspace/to_splat.py /workspace/run.sh /workspace/room_clean.png .
echo "Setup done."
