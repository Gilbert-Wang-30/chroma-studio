"""Paths, device and resolution policy. Import this instead of hardcoding anything."""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR = os.path.join(ROOT, "data")
JOBS_DIR = os.path.join(DATA_DIR, "jobs")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
PALETTE_CACHE_DIR = os.path.join(CACHE_DIR, "palettes")
SAMPLES_DIR = os.path.join(ROOT, "samples")
WEB_DIR = os.path.join(ROOT, "web")

SAM2_CHECKPOINT = os.path.join(MODELS_DIR, "sam2.1_hiera_large.pt")
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"   # resolved inside the sam2 package
INTRINSIC_VERSION = "v2.1"

# Analysis (segmentation + intrinsic) runs at this long side; the full-resolution
# original is kept for export. 1536 keeps SAM 2 crops meaningful and intrinsic under 5 GB.
WORK_LONG_SIDE = 1536
# Interactive previews are rendered at this long side so a mapping change feels instant.
PREVIEW_LONG_SIDE = 1024
# Above this many pixels the full-res intrinsic pass is skipped in favour of guided
# upsampling of the working-res layers (13.8 GB VRAM was measured at 3072x2048).
FULLRES_INTRINSIC_MAX_PIXELS = 12_000_000
# Uploads larger than this long side are downscaled on ingest.
MAX_INGEST_LONG_SIDE = 6000

SERVER_PORT = int(os.environ.get("RECOLOR_PORT", "8810"))
# SAM 2 and the intrinsic model are unloaded from the GPU after this many seconds with
# no analysis or full-resolution export in progress, then reloaded lazily on the next
# one. 0 disables idle-unload (once loaded, a model stays resident).
IDLE_UNLOAD_S = float(os.environ.get("RECOLOR_IDLE_UNLOAD_S", "120"))


def device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dirs() -> None:
    for d in (MODELS_DIR, DATA_DIR, JOBS_DIR, CACHE_DIR, PALETTE_CACHE_DIR):
        os.makedirs(d, exist_ok=True)
