#!/usr/bin/env bash
# One-shot environment setup. Idempotent. Assumes the base conda python has torch+CUDA.
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv --system-site-packages .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
.venv/bin/pip install -q --no-build-isolation "git+https://github.com/facebookresearch/sam2.git"
.venv/bin/pip install -q --no-deps \
  "altered_midas @ git+https://github.com/CCareaga/MiDaS@fb51e3a" \
  "chrislib @ git+https://github.com/CCareaga/chrislib@9a4c63f" \
  "intrinsic @ https://github.com/compphoto/Intrinsic/archive/main.zip"
mkdir -p models ~/.cache/torch/hub/checkpoints
[ -s models/sam2.1_hiera_large.pt ] || curl -L -o models/sam2.1_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
for s in 0 1 2 3 4; do f=stage_${s}_v21.pt
  [ -s ~/.cache/torch/hub/checkpoints/$f ] || curl -L -o ~/.cache/torch/hub/checkpoints/$f \
    https://github.com/compphoto/Intrinsic/releases/download/v2.1/$f
done
# Torch Hub asks interactively before fetching backbone repos; pre-trust them.
printf 'facebookresearch_WSL-Images\nrwightman_gen-efficientnet-pytorch\n' >> ~/.cache/torch/hub/trusted_list
sort -u ~/.cache/torch/hub/trusted_list -o ~/.cache/torch/hub/trusted_list
echo "ok"
