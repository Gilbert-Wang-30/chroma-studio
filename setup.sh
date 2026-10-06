#!/usr/bin/env bash
# One-shot environment setup. Idempotent. Assumes the base conda python has torch+CUDA.
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv --system-site-packages .venv
.venv/bin/pip install -q --upgrade pip
# Never let a requirement replace the base environment's torch / torchvision / numpy / opencv.
.venv/bin/python - > .venv/constraints.txt <<'EOF'
import importlib.metadata as md
for name in ("torch", "torchvision", "numpy", "opencv-python-headless"):
    try:
        print(f"{name}=={md.version(name)}")
    except md.PackageNotFoundError:
        pass
EOF
.venv/bin/pip install -q -c .venv/constraints.txt -r requirements.txt
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
# ViTMatte-small for the analysis-time edge snap (~100 MB, Apache 2.0), the snapshot it was tuned on.
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('hustvl/vitmatte-small-composition-1k', revision='6a58ad7646403c1df626fbd746900aec7361ea1d', allow_patterns=['*.json', '*.safetensors'])"
# Florence-2-large for lettering and named parts (~1.5 GB, MIT) and BiRefNet_dynamic for the foreground
# matte (~425 MB, MIT), the pinned snapshots the regions stage was measured with. Both are read from
# this cache only; the analysis never downloads.
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('florence-community/Florence-2-large', revision='4271c66b88cdbc05735372ec13b2360108de5317', allow_patterns=['*.json', '*.safetensors', '*.txt'])"
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('ZhengPeng7/BiRefNet_dynamic', revision='280306042f57b7a33854319da62fd86aaa89ec4c', allow_patterns=['*.json', '*.safetensors', '*.py'])"
# OWLv2 (large, ensemble; ~1.7 GB, Apache 2.0) for the detected parts (a shock spring, the rims, a
# grille get groups of their own), the snapshot the part gates were measured with; cache only.
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('google/owlv2-large-patch14-ensemble', revision='95e26936e865f87db1742128404b3c035d47d89d', allow_patterns=['*.json', '*.safetensors', '*.txt'])"
# Torch Hub asks interactively before fetching backbone repos; pre-trust them.
printf 'facebookresearch_WSL-Images\nrwightman_gen-efficientnet-pytorch\n' >> ~/.cache/torch/hub/trusted_list
sort -u ~/.cache/torch/hub/trusted_list -o ~/.cache/torch/hub/trusted_list
echo "ok"
