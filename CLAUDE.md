# Recolor — project conventions

Photorealistic recoloring of product photos (cars, bikes, sneakers, model kits, anything)
via a modular vision pipeline: segmentation → intrinsic decomposition → prompt palette →
color mapping → recoloring engine. See `docs/ARCHITECTURE.md` for the full contract.

## Environment

- Python: **always** the project's `.venv/bin/python` (a venv layered on the
  base conda torch 2.14+cu130 via `--system-site-packages`). Never use bare `python3`.
- GPU: one RTX 5090 (32 GB). CUDA is available; use it for every heavy step.
- Models live in `models/` (SAM 2.1 hiera-large) and `~/.cache/torch/hub/checkpoints/`
  (Intrinsic v2.1 stages + backbones). Torch Hub trusted list is pre-populated; never
  prompt interactively.
- Installed and importable: `torch`, `torchvision`, `cv2` (opencv-python-headless 5.x —
  **no GUI, no `cv2.ximgproc`**), `skimage` 0.26, `sklearn`, `scipy`, `PIL`, `sam2`,
  `intrinsic` (+ `chrislib`, `altered_midas`), `fastapi`, `uvicorn`, `python-multipart`,
  `ddgs`, `pytest`, `kornia`, `timm`.
- Sample images: `samples/*.jpg` (see `samples/MANIFEST.json`). Use them for smoke tests.
- Scratch output for experiments goes in `scratch/` (git-ignored), never in the package.

## Code style

- Python 3.13, type hints on public functions, dataclasses for records, no global mutable
  state except explicit lazy singletons for models.
- Every public function in `recolor/` has a docstring saying what it guarantees.
- Numerics: images in memory are `np.ndarray` **RGB** (never BGR) `uint8 HxWx3` for
  display, `float32 HxWx3` **linear** for intrinsic quantities. Convert at the edges only,
  using `recolor.imageio`.
- Label maps are `np.int32 HxW`; `-1` means unassigned and must not survive a stage.
- Tests: `pytest tests/` — fast, no network, no model downloads. Model-backed checks
  live in `scripts/` and are run manually.
- Frontend: **no build step**. Plain HTML + CSS + ES modules under `web/`, served by the
  FastAPI app. No npm, no bundler, no framework. Google Fonts are allowed.
- Keep files focused; do not create files outside your owned paths when working as a
  parallel build agent (ownership is listed in `docs/ARCHITECTURE.md`).

## Running

```bash
.venv/bin/python serve.py          # app on http://0.0.0.0:8810 (LAN + Tailscale printed)
.venv/bin/python -m pytest tests/  # unit tests
```

The user browses from another machine on the LAN or over Tailscale, so servers bind
`0.0.0.0` and print the LAN and Tailscale URLs, never just localhost.
