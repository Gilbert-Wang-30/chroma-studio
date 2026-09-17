"""SAM 2.1 (hiera-large) automatic mask proposals with detail presets.

`SamMasker` wraps `SAM2AutomaticMaskGenerator` as a lazy singleton: the model is
loaded on first use and shared by every instance. Presets trade time for recall on busy
scenes; the region hierarchy (`hierarchy.build_regions`) turns the overlapping proposals
into a clean partition afterwards.

Measured on an RTX 5090 shared with other jobs, 1536-px working images, bf16 autocast,
SAM only (model load ~1 s excluded); the region hierarchy adds ~0.8-1.2 s on top:

| preset   | points/side | crops | m2m | motorcycle | red car | sprue | street | interior |
|----------|-------------|-------|-----|-----------:|--------:|------:|-------:|---------:|
| fast     | 32          | 1     | no  |      0.7 s |   0.7 s | 0.5 s |  0.5 s |    0.6 s |
| balanced | 40          | 1+4   | yes |      5.1 s |   5.5 s | 4.4 s |  5.0 s |    4.8 s |
| max      | 64          | 1+4+16| yes |     18.9 s |  19.1 s |18.9 s | 18.4 s |   20.1 s |

Those numbers were taken with the GPU otherwise idle. Under contention (other agents'
jobs on the same 5090) a later run measured fast 0.6-2.6 s, balanced 5-6.6 s, and max
26.2 s on street_complex_1 (1536x1024, 1.6 MP) and 27.5 s on gundam_rx78_rg (1500x1500,
2.25 MP): the slow case for `max` is a 1536-long-side image above ~1.8 MP (square-ish
frames), which is still under the ~40 s budget but not by the margin the table suggests.

Proposal counts at max: 400-800 per image. Peak VRAM is 4.5 GB (fast) to 7.8 GB (max)
with points_per_batch 64; 256 would be only ~10 % faster but peaks at 16 GB, which is
too much for a GPU shared with the intrinsic model and the renderer.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional

import numpy as np
import torch

from .. import config

Progress = Optional[Callable[[float, str], None]]

# The levers, per preset. `min_mask_region_area` (pixels) is applied by this module
# (holes filled, islands dropped) because the sam2 CUDA extension that would do it inside
# the predictor is not built in this environment.
DETAIL_PRESETS: dict[str, dict[str, Any]] = {
    "fast": {
        "points_per_side": 32, "points_per_batch": 64,
        "crop_n_layers": 0, "crop_n_points_downscale_factor": 1,
        "pred_iou_thresh": 0.80, "stability_score_thresh": 0.92,
        "min_mask_region_area": 120, "use_m2m": False,
    },
    "balanced": {
        "points_per_side": 40, "points_per_batch": 64,
        "crop_n_layers": 1, "crop_n_points_downscale_factor": 2,
        "pred_iou_thresh": 0.76, "stability_score_thresh": 0.90,
        "min_mask_region_area": 64, "use_m2m": True,
    },
    "max": {
        "points_per_side": 64, "points_per_batch": 64,
        "crop_n_layers": 2, "crop_n_points_downscale_factor": 2,
        "pred_iou_thresh": 0.70, "stability_score_thresh": 0.88,
        "min_mask_region_area": 32, "use_m2m": True,
    },
}

_MASK_KEYS = ("segmentation", "area", "bbox", "predicted_iou", "stability_score")


def _import_sam2():
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2
    from sam2.utils.amg import MaskData, generate_crop_boxes
    return SAM2AutomaticMaskGenerator, build_sam2, MaskData, generate_crop_boxes


def _make_generator_class():
    """Subclass of SAM2AutomaticMaskGenerator that reports progress per crop.

    Built lazily so importing this module never imports sam2 (tests stay model-free).
    """
    SAM2AutomaticMaskGenerator, _, MaskData, generate_crop_boxes = _import_sam2()
    from torchvision.ops.boxes import batched_nms, box_area

    class ProgressMaskGenerator(SAM2AutomaticMaskGenerator):
        progress: Progress = None
        label: str = ""

        def _generate_masks(self, image: np.ndarray):
            orig_size = image.shape[:2]
            crop_boxes, layer_idxs = generate_crop_boxes(orig_size, self.crop_n_layers, self.crop_overlap_ratio)
            n_crops = len(crop_boxes)
            n_layers = self.crop_n_layers + 1
            data = MaskData()
            for i, (crop_box, layer_idx) in enumerate(zip(crop_boxes, layer_idxs)):
                if self.progress is not None:
                    self.progress(i / max(1, n_crops),
                                  f"{self.label} · crop {i + 1}/{n_crops} (layer {layer_idx + 1}/{n_layers})")
                data.cat(self._process_crop(image, crop_box, layer_idx, orig_size))
            if n_crops > 1:
                scores = (1 / box_area(data["crop_boxes"])).to(data["boxes"].device)
                keep = batched_nms(data["boxes"].float(), scores, torch.zeros_like(data["boxes"][:, 0]),
                                   iou_threshold=self.crop_nms_thresh)
                data.filter(keep)
            data.to_numpy()
            return data

    return ProgressMaskGenerator


def _clean_mask(seg: np.ndarray, bbox: list, min_region: int) -> tuple[np.ndarray, int, list[int]] | None:
    """Fill holes and drop islands smaller than `min_region` pixels (work is done on the
    bbox crop, so this is cheap even for thousands of masks). Returns
    (segmentation, area, bbox xywh) or None if nothing is left."""
    from sam2.utils.amg import remove_small_regions

    h, w = seg.shape
    x, y, bw, bh = (int(round(v)) for v in bbox)
    x0, y0 = max(0, x - 1), max(0, y - 1)
    x1, y1 = min(w, x + bw + 2), min(h, y + bh + 2)
    crop = seg[y0:y1, x0:x1]
    if min_region > 0 and crop.size:
        crop, ch = remove_small_regions(crop, min_region, mode="holes")
        crop, ci = remove_small_regions(crop, min_region, mode="islands")
        if ch or ci:
            seg = np.zeros_like(seg)
            seg[y0:y1, x0:x1] = crop
    ys, xs = np.nonzero(crop)
    if ys.size == 0:
        return None
    area = int(ys.size)
    bx0, bx1 = int(xs.min()) + x0, int(xs.max()) + x0
    by0, by1 = int(ys.min()) + y0, int(ys.max()) + y0
    return seg, area, [bx0, by0, bx1 - bx0 + 1, by1 - by0 + 1]


class SamMasker:
    """Lazy singleton around SAM 2.1 hiera-large (`config.SAM2_CHECKPOINT`).

    Any number of `SamMasker()` instances share one model and one generator per preset;
    `SamMasker.instance()` returns the canonical one. `generate` is serialised by a lock
    so the GPU is never asked for two proposal runs at once.
    """

    _lock = threading.Lock()
    _model: Any = None
    _model_device: str | None = None
    _generators: dict[str, Any] = {}
    _shared: "SamMasker | None" = None

    def __init__(self, device: str | None = None) -> None:
        self.device = device or config.device()

    @classmethod
    def instance(cls) -> "SamMasker":
        """The process-wide masker (created on first call)."""
        with cls._lock:
            if cls._shared is None:
                cls._shared = cls()
            return cls._shared

    # ------------------------------------------------------------ model lifecycle

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._model is not None

    def load(self) -> None:
        """Load SAM 2.1 onto `self.device` (idempotent, thread-safe). ~2 s from disk."""
        with self._lock:
            if SamMasker._model is not None:
                return
            _, build_sam2, _, _ = _import_sam2()
            model = build_sam2(config.SAM2_CONFIG, config.SAM2_CHECKPOINT, device=self.device)
            model.eval()
            SamMasker._model = model
            SamMasker._model_device = self.device

    def release(self) -> None:
        """Drop the model and every generator; frees the CUDA memory they held."""
        with self._lock:
            SamMasker._generators.clear()
            SamMasker._model = None
            SamMasker._model_device = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _generator(self, detail: str, params: dict[str, Any]):
        key = detail if params is DETAIL_PRESETS.get(detail) else repr(sorted(params.items()))
        gen = SamMasker._generators.get(key)
        if gen is None:
            cls = _make_generator_class()
            kw = {k: v for k, v in params.items() if k != "min_mask_region_area"}
            gen = cls(SamMasker._model, min_mask_region_area=0, output_mode="binary_mask", **kw)
            SamMasker._generators[key] = gen
        return gen

    # ------------------------------------------------------------ inference

    def generate(self, image_rgb_u8: np.ndarray, detail: str = "balanced",
                 progress: Progress = None, overrides: dict[str, Any] | None = None) -> list[dict]:
        """Mask proposals for an RGB uint8 image.

        Each element is `{"segmentation": bool HxW, "area": int, "bbox": [x, y, w, h],
        "predicted_iou": float, "stability_score": float}`. Masks overlap freely; holes
        and islands smaller than the preset's `min_mask_region_area` are removed, and the
        list is sorted by area descending. `overrides` patches preset keys for
        experiments. `progress(fraction, message)` is called per crop when given.
        """
        if detail not in DETAIL_PRESETS:
            raise ValueError(f"unknown detail preset {detail!r}; choose from {sorted(DETAIL_PRESETS)}")
        if image_rgb_u8.dtype != np.uint8 or image_rgb_u8.ndim != 3 or image_rgb_u8.shape[2] != 3:
            raise ValueError("generate expects an HxWx3 uint8 RGB image")
        params = DETAIL_PRESETS[detail]
        if overrides:
            params = {**params, **overrides}
        self.load()
        label = f"Finding parts with SAM 2 · {params['points_per_side']} points/side"
        with self._lock:
            gen = self._generator(detail, params)
            gen.progress = progress
            gen.label = label
            t0 = time.perf_counter()
            use_cuda = self.device.startswith("cuda") and torch.cuda.is_available()
            try:
                with torch.inference_mode():
                    if use_cuda:
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            raw = gen.generate(np.ascontiguousarray(image_rgb_u8))
                    else:
                        raw = gen.generate(np.ascontiguousarray(image_rgb_u8))
            finally:
                gen.progress = None
        min_region = int(params.get("min_mask_region_area", 0))
        out: list[dict] = []
        for m in raw:
            cleaned = _clean_mask(np.asarray(m["segmentation"], dtype=bool), m["bbox"], min_region)
            if cleaned is None:
                continue
            seg, area, bbox = cleaned
            out.append({
                "segmentation": seg, "area": area, "bbox": bbox,
                "predicted_iou": float(m["predicted_iou"]),
                "stability_score": float(m["stability_score"]),
            })
        out.sort(key=lambda d: -d["area"])
        if progress is not None:
            progress(1.0, f"{label} · {len(out)} proposals in {time.perf_counter() - t0:.1f} s")
        return out


def warmup() -> None:
    """Load SAM 2.1 onto the GPU now (idempotent); the server calls this at start."""
    SamMasker.instance().load()


def release() -> None:
    """Drop the shared model and every cached generator, freeing the CUDA memory they
    held. Idempotent; the next `generate()` call reloads from disk (~2 s)."""
    SamMasker.instance().release()


def is_loaded() -> bool:
    return SamMasker.is_loaded()
