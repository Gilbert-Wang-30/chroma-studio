"""Boundary snap of the colour groups at analysis time: ViTMatte alpha mattes, or a colour
guided filter when ViTMatte is unavailable.

A SAM label routinely stops 1-5 px short of a painted part's real edge, and those pixels
keep the old paint after a repaint (a yellow sliver along the BMW's fairing seam). Each
chromatic group is matted with ViTMatte-small (Yao et al. 2023) in a trimap band of
BAND_PX pixels each side of its label; a band pixel whose alpha reaches ALPHA_THRESHOLD
joins the group (it takes the label of the group's nearest region). Pixels only ever move
*into* a chromatic group, and they come from whichever groups sit in its band: neutral
parts (a cable, a spoke, a mirror stalk) and, where two chromatic groups meet, the other
chromatic group (the higher alpha wins a pixel both claim). Neutral groups never grow;
carved decal islands are never claimed. Measured on the two motorcycles: black-target
halo -39 % (BMW) and -8 % (Ducati), no region lost half its area.

The model is a lazy singleton like SAM 2 and the intrinsic model (about 0.5 s to load,
100 MB of weights, 1.5 GB peak for a 1536 px image, ~30 ms per group) and is released by
the pipeline's idle watchdog. The weights are read from the local Hugging Face cache only
(``setup.sh`` downloads the pinned snapshot); the analysis never goes to the network. When
``transformers`` or the weights are missing, :func:`snap_labels` falls back to snapping
every label to the photograph with a colour guided filter (radius 2), a much smaller gain,
and logs a warning once (missing weights are looked for again on the next job); when the
shared card has no room for it (CUDA OOM), or the model fails on an unusual image (a
panorama a few pixels tall), that one job uses the fallback and the next tries ViTMatte
again.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional, Sequence

import cv2
import numpy as np

from .. import filters
from ..types import ColorGroup

log = logging.getLogger("recolor.segmentation.matting")

MODEL_ID = "hustvl/vitmatte-small-composition-1k"
#: The Hugging Face snapshot the thresholds were measured with.
MODEL_REVISION = "6a58ad7646403c1df626fbd746900aec7361ea1d"
#: Trimap band (px) on each side of a group's label.
BAND_PX = 6
#: A band pixel with at least this alpha joins the group.
ALPHA_THRESHOLD = 0.5
#: Groups that are matted: non-background, CIELAB chroma of the group albedo at least this.
SNAP_CHROMA = 18.0
#: Colour guided-filter fallback.
GUIDED_RADIUS = 2
GUIDED_EPS = 1e-5
GUIDED_CHUNK = 16

Progress = Optional[Callable[[float, str], None]]

_lock = threading.Lock()
_runner: Optional["_ViTMatte"] = None
_state = "cold"                  # cold | loading | ready | unavailable
_unavailable_reason: Optional[str] = None
_weights_missing = False         # 'unavailable' only until the weights are in the local cache


# ---------------------------------------------------------------------- model lifecycle

class WeightsMissing(OSError):
    """The ViTMatte snapshot is not in the local Hugging Face cache (``setup.sh`` downloads
    it). The analysis never downloads it: that ran inside the groups stage, under the GPU
    lock, where an offline or firewalled machine blocked every GPU user for the length of
    the network timeouts."""


#: The snapshot's files that must be in the local cache before the model is loaded.
SNAPSHOT_FILES = ("config.json", "preprocessor_config.json", "model.safetensors")


def _snapshot_cached() -> bool:
    """True when the pinned snapshot is in the local cache (:mod:`hfcache`, no network)."""
    from .hfcache import snapshot_present
    return snapshot_present(MODEL_ID, MODEL_REVISION, SNAPSHOT_FILES)


class _ViTMatte:
    """ViTMatte-small on the GPU (fp16 on CUDA). ``matte`` is the only method. Loads from
    the local cache only (:class:`WeightsMissing` otherwise, raised before ``from_pretrained``
    is called when the snapshot is missing: that call asks the network for its error message
    even with ``local_files_only``)."""

    def __init__(self) -> None:
        if not _snapshot_cached():
            raise WeightsMissing(f"{MODEL_ID}@{MODEL_REVISION[:7]} is not in the local Hugging Face cache; "
                                 f"run setup.sh to download it")
        import torch
        from transformers import VitMatteForImageMatting, VitMatteImageProcessor
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        kwargs = {"revision": MODEL_REVISION, "local_files_only": True}
        try:
            self.processor = VitMatteImageProcessor.from_pretrained(MODEL_ID, **kwargs)
            model = VitMatteForImageMatting.from_pretrained(MODEL_ID, **kwargs)
        except OSError as e:
            raise WeightsMissing(f"{MODEL_ID}@{MODEL_REVISION[:7]} is not in the local Hugging Face cache; "
                                 f"run setup.sh to download it ({type(e).__name__})") from e
        self.model = model.to(self.device).eval()
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        if self.dtype == torch.float16:
            self.model.half()

    def matte(self, image_u8: np.ndarray, trimap_u8: np.ndarray) -> np.ndarray:
        """float32 HxW alpha in [0,1], exactly 1 / 0 in the trimap's known regions."""
        torch = self.torch
        h, w = trimap_u8.shape
        with torch.inference_mode():
            # channels_last is explicit: on an image 1 or 3 px tall the processor's guess of
            # the channel axis is ambiguous and it concatenates the wrong axes.
            inputs = self.processor(images=image_u8, trimaps=trimap_u8, return_tensors="pt",
                                    input_data_format="channels_last")
            pv = inputs["pixel_values"].to(self.device, self.dtype)
            out = self.model(pixel_values=pv).alphas[0, 0, :h, :w].float()
            if not bool(torch.isfinite(out).all()):
                # fp16 overflow: redo this one in fp32
                self.model.float()
                try:
                    out = self.model(pixel_values=pv.float()).alphas[0, 0, :h, :w].float()
                finally:
                    if self.dtype == torch.float16:
                        self.model.half()
        a = out.clamp(0.0, 1.0).cpu().numpy().astype(np.float32)
        a[trimap_u8 == 255] = 1.0
        a[trimap_u8 == 0] = 0.0
        return a


def _is_cuda_oom(e: BaseException) -> bool:
    return type(e).__name__ == "OutOfMemoryError" or "CUDA out of memory" in str(e)


def _get_runner() -> Optional[_ViTMatte]:
    """The loaded model, loading it on first use; None when ViTMatte cannot be loaded, in
    which case the guided-filter fallback is used. A missing package makes it
    'unavailable' for the process; missing weights make it 'unavailable' too, but the local
    cache is looked at again on the next job (so running setup.sh needs no restart; that
    check never touches the network). Either is logged once. A CUDA OOM on the shared card
    only skips it for this call."""
    global _runner, _state, _unavailable_reason, _weights_missing
    with _lock:
        if _runner is not None:
            return _runner
        if _state == "unavailable" and not _weights_missing:
            return None
        before = _state
        _state = "loading"
        t0 = time.perf_counter()
        try:
            _runner = _ViTMatte()
        except Exception as e:  # noqa: BLE001 - missing package, missing weights, CUDA trouble
            if _is_cuda_oom(e):
                _state = "cold"
                log.warning("ViTMatte did not fit on the GPU (%s); guided-filter snap for this job", type(e).__name__)
                _empty_cache()
                return None
            _state = "unavailable"
            _weights_missing = isinstance(e, WeightsMissing)
            _unavailable_reason = f"{type(e).__name__}: {e}"
            if before != "unavailable":
                log.warning("ViTMatte unavailable (%s); snapping group edges with the colour guided filter instead",
                            _unavailable_reason)
            return None
        _weights_missing = False
        _state = "ready"
        log.info("ViTMatte loaded in %.1f s", time.perf_counter() - t0)
        return _runner


def _empty_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def status() -> str:
    """'cold' | 'loading' | 'ready' | 'unavailable' (no ``transformers`` or no weights)."""
    return _state


def is_loaded() -> bool:
    """True while the model holds GPU memory."""
    return _runner is not None


def warmup() -> None:
    """Load the model now (idempotent). A missing package or missing weights leave the
    module 'unavailable'; a CUDA OOM leaves it 'cold'."""
    _get_runner()


def release() -> None:
    """Drop the model and free its CUDA memory (idempotent). It reloads lazily on the next
    analysis; an 'unavailable' state is kept, so a missing package is not retried."""
    global _runner, _state
    with _lock:
        if _runner is None:
            return
        _runner = None
        _state = "cold"
    _empty_cache()


# ---------------------------------------------------------------------- trimap and label move

def _disk(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def make_trimap(hard: np.ndarray, band_px: int = BAND_PX) -> np.ndarray:
    """uint8 HxW trimap of a group's label: 255 = sure group (label eroded by ``band_px``),
    0 = sure not (outside the label dilated by ``band_px``), 128 = the unknown band. A
    connected part too thin to survive the erosion keeps a core from a smaller erosion,
    so no part is ever without a foreground anchor."""
    h = (np.asarray(hard) > 0.5).astype(np.uint8)
    fg = cv2.erode(h, _disk(band_px)) if band_px > 0 else h.copy()
    n, cc = cv2.connectedComponents(h, connectivity=8)
    if n > 1:
        survived = np.zeros(n, bool)
        survived[np.unique(cc[fg > 0])] = True
        lost = ~survived
        lost[0] = False
        if lost.any():
            lost_mask = lost[cc].astype(np.uint8)
            r = band_px - 1
            while r >= 0:
                e = cv2.erode(lost_mask, _disk(r)) if r > 0 else lost_mask
                fg = np.maximum(fg, e)
                still = lost.copy()
                still[np.unique(cc[e > 0])] = False
                still[0] = False
                if not still.any():
                    break
                lost_mask = still[cc].astype(np.uint8)
                r -= 1
    dil = cv2.dilate(h, _disk(band_px)) if band_px > 0 else h
    tri = np.zeros(h.shape, np.uint8)
    tri[dil > 0] = 128
    tri[fg > 0] = 255
    return tri


def _nearest_label(labels: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """For every pixel, the label of the nearest pixel where ``allowed`` is True."""
    from scipy import ndimage
    idx = ndimage.distance_transform_edt(~allowed, return_distances=False, return_indices=True)
    return labels[idx[0], idx[1]]


def matte_move_labels(labels: np.ndarray, group_map: np.ndarray, alphas: dict[int, np.ndarray],
                      bands: dict[int, np.ndarray], protect: Optional[np.ndarray] = None,
                      threshold: float = ALPHA_THRESHOLD) -> np.ndarray:
    """Grow each group of ``alphas`` into its unknown band where its alpha >= ``threshold``
    (the label of the group's nearest region); competing claims go to the higher alpha
    (the first group on ties); ``protect`` pixels are never claimed. A claimed pixel leaves
    whichever group it was in (a neutral part or another matted group); groups without an
    alpha never grow."""
    shape = labels.shape
    best = np.full(shape, -1.0, np.float32)
    claim = np.full(shape, -1, np.int32)
    for gid, a in alphas.items():
        c = bands[gid] & (a >= threshold) & (group_map != gid)
        if protect is not None:
            c &= ~protect
        take = c & (a > best)
        best[take] = a[take]
        claim[take] = gid
    out = labels.copy()
    for gid in alphas:
        sel = claim == gid
        if sel.any():
            out[sel] = _nearest_label(labels, group_map == gid)[sel]
    return out


# ---------------------------------------------------------------------- guided-filter fallback

def guided_snap(labels: np.ndarray, guide_rgb01: np.ndarray, protect: Optional[np.ndarray] = None,
                radius: int = GUIDED_RADIUS, eps: float = GUIDED_EPS) -> np.ndarray:
    """Every label's indicator filtered with the full colour guided filter of the photo;
    each pixel takes the label with the highest response (it can only move to a label
    present within about 2 * radius). ``protect`` pixels keep their label and no other
    pixel may take one of their labels, so carved islands stay exactly as they are."""
    import torch
    dev = filters._dev()
    ids = np.unique(labels)
    if len(ids) <= 1 or min(labels.shape) <= 2 * radius + 1:
        return labels.copy()
    g = torch.from_numpy(np.ascontiguousarray(guide_rgb01, np.float32)).permute(2, 0, 1).contiguous().to(dev)
    box = filters.box_filter
    mI = box(g, radius)
    II = box(torch.stack((g[0] * g[0], g[0] * g[1], g[0] * g[2], g[1] * g[1], g[1] * g[2], g[2] * g[2])), radius)
    rr = II[0] - mI[0] * mI[0] + eps
    rg = II[1] - mI[0] * mI[1]
    rb = II[2] - mI[0] * mI[2]
    gg = II[3] - mI[1] * mI[1] + eps
    gb = II[4] - mI[1] * mI[2]
    bb = II[5] - mI[2] * mI[2] + eps
    c0, c1, c2 = gg * bb - gb * gb, rb * gb - rg * bb, rg * gb - rb * gg
    det = rr * c0 + rg * c1 + rb * c2
    det = torch.where(det.abs() < 1e-12, torch.full_like(det, 1e-12), det)
    i11, i12, i13 = c0 / det, c1 / det, c2 / det
    i22, i23, i33 = (rr * bb - rb * rb) / det, (rb * rg - rr * gb) / det, (rr * gg - rg * rg) / det
    lab_t = torch.from_numpy(labels.astype(np.int32)).to(dev)
    best = torch.full(labels.shape, -1e9, device=dev)
    out = lab_t.clone()
    for i in range(0, len(ids), GUIDED_CHUNK):
        sel = torch.from_numpy(ids[i:i + GUIDED_CHUNK].astype(np.int32)).to(dev)
        p = (lab_t[None] == sel[:, None, None]).float()
        mp = box(p, radius)
        cov0 = box(g[0][None] * p, radius) - mI[0][None] * mp
        cov1 = box(g[1][None] * p, radius) - mI[1][None] * mp
        cov2 = box(g[2][None] * p, radius) - mI[2][None] * mp
        a0 = i11 * cov0 + i12 * cov1 + i13 * cov2
        a1 = i12 * cov0 + i22 * cov1 + i23 * cov2
        a2 = i13 * cov0 + i23 * cov1 + i33 * cov2
        b = mp - (a0 * mI[0] + a1 * mI[1] + a2 * mI[2])
        resp = box(a0, radius) * g[0] + box(a1, radius) * g[1] + box(a2, radius) * g[2] + box(b, radius)
        m, arg = resp.max(0)
        better = m > best
        best = torch.where(better, m, best)
        out = torch.where(better, sel[arg], out)
    res = out.cpu().numpy().astype(np.int32)
    if protect is not None and protect.any():
        kept = np.unique(labels[protect])
        res = np.where(protect | np.isin(res, kept), labels, res)
    return res


# ---------------------------------------------------------------------- public entry point

def _chroma(g: ColorGroup) -> float:
    return float(np.hypot(g.albedo_lab[1], g.albedo_lab[2]))


def snap_labels(image_rgb_u8: np.ndarray, labels: np.ndarray, group_map: np.ndarray,
                groups: Sequence[ColorGroup], protect: Optional[np.ndarray] = None,
                progress: Progress = None) -> tuple[np.ndarray, str]:
    """Move the label boundaries of the chromatic groups onto the photograph's edges.

    Returns ``(labels, method)``: an int32 label map with the same ids (a region may shrink
    or grow; the caller rebuilds the records, see ``refine.relabel``) and ``'vitmatte'``,
    ``'guided'`` (fallback) or ``'none'`` (no chromatic group). ``protect`` pixels (decal
    islands) never change label."""
    labels = np.ascontiguousarray(labels, np.int32)
    chrom = [g for g in groups if not g.is_background and _chroma(g) >= SNAP_CHROMA]
    if not chrom:
        return labels.copy(), "none"
    runner = _get_runner()
    if runner is not None:
        alphas: dict[int, np.ndarray] = {}
        bands: dict[int, np.ndarray] = {}
        try:
            for k, g in enumerate(chrom):
                tri = make_trimap(group_map == g.id)
                alphas[g.id] = runner.matte(image_rgb_u8, tri)
                bands[g.id] = tri == 128
                if progress is not None:
                    progress((k + 1) / len(chrom), f"Snapping paint edges with ViTMatte · {k + 1}/{len(chrom)}")
        except Exception as e:  # noqa: BLE001 - the fallback exists; never fail the analysis here
            if _is_cuda_oom(e):
                log.warning("ViTMatte ran out of GPU memory; guided-filter snap for this job")
                release()
            else:
                log.warning("ViTMatte failed on this image (%s: %s); guided-filter snap for this job",
                            type(e).__name__, e)
                _empty_cache()
        else:
            return matte_move_labels(labels, group_map, alphas, bands, protect), "vitmatte"
    if progress is not None:
        progress(0.5, "Snapping paint edges (guided filter)")
    return guided_snap(labels, image_rgb_u8.astype(np.float32) / 255.0, protect), "guided"
