"""OWLv2 at analysis time: open-vocabulary boxes of the parts people personalise.

The regions stage asks OWLv2 (Minderer et al. 2023, ``google/owlv2-large-patch14-ensemble``,
Apache 2.0) for the phrases of the photo's part vocabulary (:mod:`smallparts`: "coil spring",
"handlebar grip", "wheel rim", "front grille" ...) on the work image and its four 0.6 x 0.6
corner tiles, every phrase in one pass per crop, and hands the boxes to
:func:`smallparts.find_kind_parts`, which prompts them as SAM boxes and gates the masks. The
tiles see a small part (a grip, a footpeg) at 1.7x the scale of the full image.

Measured on the ten-photo part reference set against Florence-2's open-vocabulary detection
and Grounding DINO: Florence-2 answers most phrases with a tile-sized box and its token
confidences do not separate right from wrong; Grounding DINO is best on wheels and seats but
misses the small parts, and its wheel boxes cost region isolation; OWLv2 finds the small parts
(the Ducati's spring 0.48, a grip 0.40, a sprocket 0.45). None of them scores a brake caliper
on the three reference bikes above 0.22 at the scale of the photo; :func:`detect_in` is the
zoomed second look :func:`smallparts.find_calipers` runs on each wheel and on each caliper
candidate (a square crop 2.5x its size: the bikes' calipers score 0.26-0.38 there).

The model is a lazy singleton like SAM 2, the intrinsic model, ViTMatte, Florence-2 and BiRefNet
(fp16 on CUDA: about 0.9 GB of VRAM; 0.2-0.5 s per 1536-px image for the five crops), released
by the pipeline's idle watchdog. The weights are read from the local Hugging Face cache only
(``setup.sh`` downloads the pinned snapshot, and :mod:`hfcache` checks for it before
``from_pretrained`` is called, which asks the network for its error message when the files
are missing); the analysis never goes to the network. Without ``transformers`` or the weights
the stage runs without detected parts (logged once; missing weights are looked for again on the
next job), and so does a job on which the model runs out of GPU memory or fails.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np

log = logging.getLogger("recolor.segmentation.partdetect")

MODEL_ID = "google/owlv2-large-patch14-ensemble"
#: The Hugging Face snapshot the gates were measured with.
MODEL_REVISION = "95e26936e865f87db1742128404b3c035d47d89d"
#: The detector name the part gates are keyed by (:class:`smallparts.PartGates`).
NAME = "owlv2"
#: The image and its 2x2 corner tiles of this share of each side.
TILE_FRAC = 0.6
#: Boxes below this score are not returned (the part gate is 0.3).
MIN_SCORE = 0.05

_lock = threading.Lock()
_runner: Optional["_Owl"] = None
_state = "cold"                  # cold | loading | ready | unavailable
_unavailable_reason: Optional[str] = None
_weights_missing = False


class WeightsMissing(OSError):
    """The OWLv2 snapshot is not in the local Hugging Face cache (``setup.sh`` downloads it)."""


#: The snapshot's files that must be in the local cache before the model is loaded.
SNAPSHOT_FILES = ("config.json", "preprocessor_config.json", "model.safetensors")


def _snapshot_cached() -> bool:
    """True when the pinned snapshot is in the local cache (:mod:`hfcache`, no network)."""
    from .hfcache import snapshot_present
    return snapshot_present(MODEL_ID, MODEL_REVISION, SNAPSHOT_FILES)


class _Owl:
    """OWLv2 on the GPU (fp16 on CUDA), loaded from the local cache only. A missing snapshot
    raises :class:`WeightsMissing` before ``from_pretrained`` is called (which asks the network
    for its error message even with ``local_files_only``)."""

    def __init__(self) -> None:
        if not _snapshot_cached():
            raise WeightsMissing(f"{MODEL_ID}@{MODEL_REVISION[:7]} is not in the local Hugging Face cache; "
                                 f"run setup.sh to download it")
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        kwargs = {"revision": MODEL_REVISION, "local_files_only": True}
        try:
            self.processor = Owlv2Processor.from_pretrained(MODEL_ID, **kwargs)
            model = Owlv2ForObjectDetection.from_pretrained(MODEL_ID, dtype=self.dtype, **kwargs)
        except OSError as e:
            raise WeightsMissing(f"{MODEL_ID}@{MODEL_REVISION[:7]} is not in the local Hugging Face cache; "
                                 f"run setup.sh to download it ({type(e).__name__})") from e
        self.model = model.to(self.device).eval()

    def _pass(self, im, queries: list[str]):
        """One forward pass on a PIL image: ``(scores [boxes, queries], boxes [boxes, 4])``,
        sigmoid scores and (cx, cy, w, h) boxes relative to the padded square, on the CPU."""
        torch = self.torch
        inputs = self.processor(text=[queries], images=im, return_tensors="pt")
        inputs = {n: (v.to(self.device, self.dtype) if v.dtype.is_floating_point else v.to(self.device))
                  for n, v in inputs.items()}
        with torch.inference_mode():
            o = self.model(**inputs)
        return o.logits[0].float().sigmoid().cpu(), o.pred_boxes[0].float().cpu()

    def boxes(self, image_u8: np.ndarray, phrases: list[str], min_score: float = MIN_SCORE,
              crops: Optional[list[tuple[int, int, int, int]]] = None, all_scores: bool = False) -> list[dict]:
        """Every box of any phrase at ``min_score`` or more, in image pixels: ``{"box": [x0, y0,
        x1, y1], "phrase", "score", "src", "det"}`` (``phrase`` the best-scoring one). By default
        on the image and its tiles; with ``crops`` on those windows only (``src`` "crop<k>"). With
        ``all_scores`` every box also carries ``scores``, the score of each phrase in order (OWLv2
        scores every query independently, so a phrase's score does not depend on the others)."""
        from PIL import Image
        h, w = image_u8.shape[:2]
        pil = Image.fromarray(np.ascontiguousarray(image_u8))
        if crops is None:
            windows = [(0, 0, w, h)] + (tiles(w, h) if min(w, h) >= 64 else [])
            names = ["full"] + [f"tile{k}" for k in range(1, len(windows))]
        else:
            windows = [tuple(int(v) for v in c) for c in crops]
            names = [f"crop{k}" for k in range(len(windows))]
        queries = [f"a photo of a {p}" for p in phrases]
        out: list[dict] = []
        for c, src in zip(windows, names):
            if c[2] - c[0] < 8 or c[3] - c[1] < 8:
                continue
            im = pil if c == (0, 0, w, h) else pil.crop(c)
            logits, pred = self._pass(im, queries)
            side = max(im.size)                               # boxes are relative to the padded square
            sc, qi = logits.max(-1)
            keep = (sc >= min_score).nonzero().flatten().tolist()
            for i in keep:
                cx, cy, bw, bh = (float(v) * side for v in pred[i].tolist())
                bx = [max(c[0], cx - bw / 2 + c[0]), max(c[1], cy - bh / 2 + c[1]),
                      min(c[2], cx + bw / 2 + c[0]), min(c[3], cy + bh / 2 + c[1])]
                if bx[2] - bx[0] < 2 or bx[3] - bx[1] < 2:
                    continue
                d = {"box": bx, "phrase": phrases[int(qi[i])], "score": float(sc[i]), "src": src, "det": NAME}
                if all_scores:
                    d["scores"] = [float(v) for v in logits[i].tolist()]
                out.append(d)
        return out


def tiles(w: int, h: int, frac: float = TILE_FRAC) -> list[tuple[int, int, int, int]]:
    """The four corner tiles (x0, y0, x1, y1) of a w x h image, ``frac`` of each side."""
    tw, th = int(round(frac * w)), int(round(frac * h))
    return [(0, 0, tw, th), (w - tw, 0, w, th), (0, h - th, tw, h), (w - tw, h - th, w, h)]


def _is_cuda_oom(e: BaseException) -> bool:
    return type(e).__name__ == "OutOfMemoryError" or "CUDA out of memory" in str(e)


def _empty_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _get_runner() -> Optional[_Owl]:
    """The loaded model, loading it on first use; None when it cannot be loaded (a missing
    package or missing weights, logged once; the local cache is looked at again on the next
    job when only the weights were missing) or when the shared card has no room for it."""
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
            _runner = _Owl()
        except Exception as e:  # noqa: BLE001 - missing package, missing weights, CUDA trouble
            if _is_cuda_oom(e):
                _state = "cold"
                log.warning("OWLv2 did not fit on the GPU (%s); no detected parts for this job", type(e).__name__)
                _empty_cache()
                return None
            _state = "unavailable"
            _weights_missing = isinstance(e, WeightsMissing)
            _unavailable_reason = f"{type(e).__name__}: {e}"
            if before != "unavailable":
                log.warning("OWLv2 unavailable (%s); the regions stage runs without detected parts",
                            _unavailable_reason)
            return None
        _weights_missing = False
        _state = "ready"
        log.info("OWLv2 loaded in %.1f s", time.perf_counter() - t0)
        return _runner


def status() -> str:
    """'cold' | 'loading' | 'ready' | 'unavailable' (no ``transformers`` or no weights)."""
    return _state


def is_loaded() -> bool:
    """True while the model holds GPU memory."""
    return _runner is not None


def warmup() -> None:
    """Load the model now (idempotent)."""
    _get_runner()


def release() -> None:
    """Drop the model and free its CUDA memory (idempotent); it reloads lazily on the next
    analysis. An 'unavailable' state is kept."""
    global _runner, _state
    with _lock:
        if _runner is None:
            return
        _runner = None
        _state = "cold"
    _empty_cache()


def _run(image_rgb_u8: np.ndarray, phrases: list[str], what: str, **kw) -> Optional[list[dict]]:
    if image_rgb_u8.dtype != np.uint8 or image_rgb_u8.ndim != 3 or image_rgb_u8.shape[2] != 3:
        raise ValueError(f"{what} expects an HxWx3 uint8 RGB image")
    if not phrases:
        return []
    runner = _get_runner()
    if runner is None:
        return None
    try:
        return runner.boxes(image_rgb_u8, list(phrases), **kw)
    except Exception as e:  # noqa: BLE001 - an enhancement, never a failure of the analysis
        if _is_cuda_oom(e):
            log.warning("OWLv2 ran out of GPU memory; no detected parts for this job")
            release()
        else:
            log.warning("OWLv2 failed on this image (%s: %s); no detected parts for this job", type(e).__name__, e)
            _empty_cache()
        return None


def detect(image_rgb_u8: np.ndarray, phrases: list[str]) -> Optional[list[dict]]:
    """Boxes of ``phrases`` on a work image (see :meth:`_Owl.boxes`), or None when the model is
    unavailable, does not fit on the shared card or fails on this image (all logged: the stage
    then runs without detected parts). An empty phrase list gives []."""
    return _run(image_rgb_u8, phrases, "detect")


def detect_in(image_rgb_u8: np.ndarray, crops: list[tuple[int, int, int, int]], phrases: list[str],
              min_score: float = MIN_SCORE) -> Optional[list[dict]]:
    """A zoomed second look: one pass per window ``crops`` ((x0, y0, x1, y1) of the work image,
    no tiles), every box of any phrase at ``min_score`` or more, in image pixels, each with
    ``src`` "crop<k>" (k the window's index) and ``scores`` (every phrase's score, in order).
    None when the model is unavailable, does not fit or fails (logged); no window gives []."""
    if not crops:
        return []
    return _run(image_rgb_u8, phrases, "detect_in", min_score=min_score, crops=list(crops), all_scores=True)
