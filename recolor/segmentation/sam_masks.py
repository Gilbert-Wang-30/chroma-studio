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

Interactive prompts (Select part, Find part: :mod:`interactive`) run on a
:class:`PromptSession` per job: the work image's embedding, computed once and kept for the
next prompts (an LRU of ``PROMPT_SESSIONS`` jobs on the shared model, dropped by
:meth:`SamMasker.release` with the model and by :meth:`SamMasker.forget_prompts` when a job
goes), plus the embedding of the last crop a mask was refined on. Measured on the 5090 with
the Ducati's 1536 x 1024 work image: the embedding 0.115 s the first time after a load and
0.023-0.029 s after that, a prompt 2.7-3.4 ms, a 200 px crop's embedding 0.022 s; 1.3 GB peak.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
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


class PromptSession:
    """One job's work image embedded for interactive prompts (:mod:`interactive`), plus the
    embedding of the last crop a mask was refined on. Made and cached by
    :meth:`SamMasker.prompt_session`; every model call takes the masker's lock, so a prompt never
    runs beside a proposal run or another prompt."""

    def __init__(self, key: str, stamp: Any, image_rgb_u8: np.ndarray, masker: "SamMasker") -> None:
        self.key = key
        self.stamp = stamp
        self.image = np.ascontiguousarray(image_rgb_u8)
        self.shape = self.image.shape[:2]
        self._masker = masker
        self._full = masker._new_predictor()
        self._crop = None
        self._crop_box: Optional[tuple[int, int, int, int]] = None
        with masker._inference():
            self._full.set_image(self.image)

    def predict(self, point_coords: Optional[np.ndarray], point_labels: Optional[np.ndarray],
                box: Optional[np.ndarray], mask_input: Optional[np.ndarray], multimask: bool,
                crop: Optional[tuple[int, int, int, int]] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """SAM's ``(masks bool [C, h, w], scores [C], low-res logits [C, 256, 256], computed)`` for
        prompts in the frame of the full image (``crop`` None) or of the window ``crop`` (x0, y0,
        x1, y1 of the work image), whose embedding is made the first time it is asked for
        (``computed``) and kept until another window is asked for."""
        with SamMasker._lock:
            computed = False
            pred = self._full
            if crop is not None:
                crop = tuple(int(v) for v in crop)
                if self._crop is None or self._crop_box != crop:
                    x0, y0, x1, y1 = crop
                    if self._crop is None:
                        self._crop = self._masker._new_predictor()
                    with self._masker._inference():
                        self._crop.set_image(np.ascontiguousarray(self.image[y0:y1, x0:x1]))
                    self._crop_box = crop
                    computed = True
                pred = self._crop
            with self._masker._inference():
                masks, scores, low = pred.predict(point_coords=point_coords, point_labels=point_labels, box=box,
                                                  mask_input=mask_input, multimask_output=bool(multimask))
        masks = np.asarray(masks) > 0.5 if np.asarray(masks).dtype != bool else np.asarray(masks)
        return masks, np.asarray(scores, np.float32).ravel(), np.asarray(low, np.float32), computed


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
    #: Interactive prompt sessions (one embedded work image each), most recently used last.
    _prompt_sessions: "OrderedDict[str, PromptSession]" = OrderedDict()
    #: How many jobs keep their embedding for interactive prompts (about 20 MB each).
    PROMPT_SESSIONS = 3

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
        """Drop the model, every generator and every prompt session's embedding; frees the CUDA
        memory they held."""
        with self._lock:
            SamMasker._generators.clear()
            SamMasker._prompt_sessions.clear()
            SamMasker._model = None
            SamMasker._model_device = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------ interactive prompts

    def _new_predictor(self):
        """A ``SAM2ImagePredictor`` on the shared model (tests replace this)."""
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        return SAM2ImagePredictor(SamMasker._model)

    def _inference(self):
        """The context every model call of a prompt runs in (inference mode, bf16 on CUDA)."""
        import contextlib
        use_cuda = self.device.startswith("cuda") and torch.cuda.is_available()
        stack = contextlib.ExitStack()
        stack.enter_context(torch.inference_mode())
        stack.enter_context(torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda))
        return stack

    def prompt_session(self, key: str, stamp: Any, load_image: Callable[[], np.ndarray]) -> tuple[PromptSession, bool]:
        """The interactive prompt session of image ``key`` (a job id) and whether its embedding
        was computed now: the cached one while its ``stamp`` (the work image's file stamp)
        matches, else ``load_image()`` is embedded once (the least recently used of more than
        ``PROMPT_SESSIONS`` sessions is dropped). Loads the model when needed."""
        self.load()
        with self._lock:
            s = SamMasker._prompt_sessions.get(key)
            if s is not None and s.stamp == stamp:
                SamMasker._prompt_sessions.move_to_end(key)
                return s, False
        image = load_image()
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("prompt sessions need an HxWx3 uint8 RGB image")
        with self._lock:
            s = SamMasker._prompt_sessions.get(key)
            if s is not None and s.stamp == stamp:             # embedded by a racing request meanwhile
                SamMasker._prompt_sessions.move_to_end(key)
                return s, False
            s = PromptSession(key, stamp, image, self)
            SamMasker._prompt_sessions[key] = s
            SamMasker._prompt_sessions.move_to_end(key)
            while len(SamMasker._prompt_sessions) > max(1, int(self.PROMPT_SESSIONS)):
                SamMasker._prompt_sessions.popitem(last=False)
        return s, True

    @classmethod
    def forget_prompts(cls, key: Optional[str] = None) -> None:
        """Drop the prompt session of ``key`` (every session when None), e.g. when its job goes."""
        with cls._lock:
            if key is None:
                cls._prompt_sessions.clear()
            else:
                cls._prompt_sessions.pop(key, None)

    @classmethod
    def prompt_keys(cls) -> list[str]:
        """The keys of the cached prompt sessions, least recently used first."""
        with cls._lock:
            return list(cls._prompt_sessions)

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


    def prompt_parts(self, image_rgb_u8: np.ndarray, points: list[tuple[int, int]],
                     crop: int = 192) -> list[list[dict]]:
        """One point prompt per ``(x, y)``, each on a ``crop`` x ``crop`` window of the image
        around it, so a small part gets SAM's full input resolution (the automatic
        generator sees it at 1/1.5 of the working image). Returns, per point, SAM's three
        candidate masks as ``{"mask": bool crop, "x0", "y0": crop offset, "score": predicted
        IoU, "clipped": the mask touches the window's border}`` (a clipped mask belongs to
        an object larger than the window). Used by `hierarchy.recover_parts`; measured
        ~40 ms per point on the 5090, 1.3 GB peak."""
        if image_rgb_u8.dtype != np.uint8 or image_rgb_u8.ndim != 3 or image_rgb_u8.shape[2] != 3:
            raise ValueError("prompt_parts expects an HxWx3 uint8 RGB image")
        if not points:
            return []
        self.load()
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        h, w = image_rgb_u8.shape[:2]
        c = int(max(16, min(crop, h, w)))
        out: list[list[dict]] = []
        use_cuda = self.device.startswith("cuda") and torch.cuda.is_available()
        with self._lock:
            pred = SAM2ImagePredictor(SamMasker._model)
            try:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda):
                    for x, y in points:
                        x0 = int(np.clip(int(x) - c // 2, 0, w - c))
                        y0 = int(np.clip(int(y) - c // 2, 0, h - c))
                        pred.set_image(np.ascontiguousarray(image_rgb_u8[y0:y0 + c, x0:x0 + c]))
                        masks, scores, _ = pred.predict(point_coords=np.array([[x - x0, y - y0]], np.float32),
                                                        point_labels=np.array([1], np.int32), multimask_output=True)
                        cands = []
                        for m, s in zip(np.asarray(masks) > 0.5, np.asarray(scores, np.float32)):
                            clipped = bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())
                            cands.append({"mask": m.astype(bool), "x0": x0, "y0": y0, "score": float(s), "clipped": clipped})
                        out.append(cands)
            finally:
                pred.reset_predictor()
        return out

    def prompt_boxes(self, image_rgb_u8: np.ndarray, jobs: list[dict]) -> list[list[dict]]:
        """Box and point prompts on crops, the way `hierarchy` and `smallparts` ask for a
        named part or a word of lettering. A job is ``{"crop": (x0, y0, x1, y1)}`` in image
        coordinates plus ``"box": [x0, y0, x1, y1]`` and / or ``"points": [[x, y], ...]`` with
        ``"labels": [1 | 0, ...]``. Jobs on the same crop share one ``set_image`` (their prompts
        are batched when they have the same layout). Returns, per job, SAM's candidate masks as
        ``{"mask": bool crop, "x0", "y0": crop offset, "score": predicted IoU, "clipped": the
        mask touches a crop side that is not the image border}`` (a clipped mask belongs to an
        object larger than the window). A crop gets SAM's full 1024-px input, so a 256-px
        window is seen at four times the automatic generator's scale."""
        if image_rgb_u8.dtype != np.uint8 or image_rgb_u8.ndim != 3 or image_rgb_u8.shape[2] != 3:
            raise ValueError("prompt_boxes expects an HxWx3 uint8 RGB image")
        out: list[list[dict]] = [[] for _ in jobs]
        if not jobs:
            return out
        self.load()
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        h, w = image_rgb_u8.shape[:2]
        by_crop: dict[tuple[int, int, int, int], list[int]] = {}
        for i, j in enumerate(jobs):
            x0, y0, x1, y1 = (int(v) for v in j["crop"])
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, max(x1, x0 + 1)), min(h, max(y1, y0 + 1))
            by_crop.setdefault((x0, y0, x1, y1), []).append(i)
        use_cuda = self.device.startswith("cuda") and torch.cuda.is_available()
        with self._lock:
            pred = SAM2ImagePredictor(SamMasker._model)
            try:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda):
                    for (x0, y0, x1, y1), idx in by_crop.items():
                        pred.set_image(np.ascontiguousarray(image_rgb_u8[y0:y1, x0:x1]))
                        layouts: dict[tuple[int, bool], list[int]] = {}
                        for i in idx:
                            j = jobs[i]
                            layouts.setdefault((len(j.get("points") or []), j.get("box") is not None), []).append(i)
                        for (npts, has_box), ii in layouts.items():
                            pc = pl = bx = None
                            if npts:
                                pc = np.array([[[p[0] - x0, p[1] - y0] for p in jobs[i]["points"]] for i in ii], np.float32)
                                pl = np.array([jobs[i]["labels"] for i in ii], np.int32)
                            if has_box:
                                bx = np.array([[jobs[i]["box"][0] - x0, jobs[i]["box"][1] - y0,
                                                jobs[i]["box"][2] - x0, jobs[i]["box"][3] - y0] for i in ii], np.float32)
                            masks, scores, _ = pred.predict(point_coords=pc, point_labels=pl, box=bx, multimask_output=True)
                            masks = np.asarray(masks) > 0.0
                            scores = np.asarray(scores, np.float32)
                            if masks.ndim == 3:                 # one job: [3, h, w] -> [1, 3, h, w]
                                masks, scores = masks[None], scores[None]
                            for k, i in enumerate(ii):
                                cands = []
                                for m, s in zip(masks[k], scores[k]):
                                    m = m.astype(bool)
                                    clipped = ((y0 > 0 and bool(m[0].any())) or (y1 < h and bool(m[-1].any()))
                                               or (x0 > 0 and bool(m[:, 0].any())) or (x1 < w and bool(m[:, -1].any())))
                                    cands.append({"mask": m, "x0": x0, "y0": y0, "score": float(s), "clipped": clipped})
                                out[i] = cands
            finally:
                pred.reset_predictor()
        return out


def warmup() -> None:
    """Load SAM 2.1 onto the GPU now (idempotent); the server calls this at start."""
    SamMasker.instance().load()


def release() -> None:
    """Drop the shared model, every cached generator and every prompt session, freeing the
    CUDA memory they held. Idempotent; the next `generate()` call reloads from disk (~2 s)."""
    SamMasker.instance().release()


def forget_prompts(key: Optional[str] = None) -> None:
    """Drop the interactive prompt session of job ``key`` (all of them when None)."""
    SamMasker.forget_prompts(key)


def is_loaded() -> bool:
    return SamMasker.is_loaded()
