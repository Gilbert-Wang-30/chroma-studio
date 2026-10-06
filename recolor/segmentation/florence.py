"""Florence-2-large at analysis time: lettering (OCR with regions) and named-part boxes.

The regions stage asks Florence-2 (Xiao et al. 2023, MIT, ``florence-community/Florence-2-large``)
two things about the work image and hands the answers to :mod:`smallparts`, which turns them
into SAM box prompts:

* ``<OCR_WITH_REGION>`` on the image and its four 0.6 x 0.6 corner tiles, batched in one
  ``generate`` call: the quads of printed words (a fairing decal, a tank logo). Greedy decoding
  gives the same quads as beam search at a quarter of the time.
* ``<CAPTION_TO_PHRASE_GROUNDING>`` with one caption of part names (PHRASES): boxes of the
  tyres, rims, discs, lamps and emblems SAM's automatic proposals return as one mask or not at
  all.

The model is a lazy singleton like SAM 2, the intrinsic model and ViTMatte (1.55 GB of fp16
weights, about 2 s to load, 1.9 GB of VRAM; 0.3-1.0 s per 1536-px image), released by the
pipeline's idle watchdog. The weights are read from the local Hugging Face cache only
(``setup.sh`` downloads the pinned snapshot); the analysis never goes to the network. When
``transformers`` or the weights are missing the stage runs without lettering and named parts
(logged once; missing weights are looked for again on the next job), and so does a job on
which the model runs out of GPU memory or fails.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

import numpy as np

log = logging.getLogger("recolor.segmentation.florence")

MODEL_ID = "florence-community/Florence-2-large"
#: The Hugging Face snapshot the pieces were measured with.
MODEL_REVISION = "4271c66b88cdbc05735372ec13b2360108de5317"
#: Part names of the grounding caption (a PACO part vocabulary subset: vehicles, bikes, small
#: hardware). One caption, one call.
PHRASES = ("tire", "wheel rim", "brake disc", "fork", "exhaust pipe", "muffler", "headlight", "turn signal",
           "mirror", "logo", "emblem", "spoke")
CAPTION = ". ".join(PHRASES) + "."
#: OCR runs on the image and on 2x2 tiles of this share of each side (20 % overlap in the middle).
TILE_FRAC = 0.6
MAX_NEW_TOKENS = 1024

_lock = threading.Lock()
_runner: Optional["_Florence"] = None
_state = "cold"                  # cold | loading | ready | unavailable
_unavailable_reason: Optional[str] = None
_weights_missing = False


class WeightsMissing(OSError):
    """The Florence-2 snapshot is not in the local Hugging Face cache (``setup.sh`` downloads it)."""


#: The snapshot's files that must be in the local cache before the model is loaded.
SNAPSHOT_FILES = ("config.json", "preprocessor_config.json", "model.safetensors")


def _snapshot_cached() -> bool:
    """True when the pinned snapshot is in the local cache (:mod:`hfcache`, no network)."""
    from .hfcache import snapshot_present
    return snapshot_present(MODEL_ID, MODEL_REVISION, SNAPSHOT_FILES)


class _Florence:
    """Florence-2-large on the GPU (fp16 on CUDA), loaded from the local cache only. A missing
    snapshot raises :class:`WeightsMissing` before ``from_pretrained`` is called (which asks the
    network for its error message even with ``local_files_only``)."""

    def __init__(self) -> None:
        if not _snapshot_cached():
            raise WeightsMissing(f"{MODEL_ID}@{MODEL_REVISION[:7]} is not in the local Hugging Face cache; "
                                 f"run setup.sh to download it")
        import torch
        from transformers import AutoProcessor, Florence2ForConditionalGeneration
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        kwargs = {"revision": MODEL_REVISION, "local_files_only": True}
        try:
            self.processor = AutoProcessor.from_pretrained(MODEL_ID, **kwargs)
            model = Florence2ForConditionalGeneration.from_pretrained(MODEL_ID, dtype=self.dtype, **kwargs)
        except OSError as e:
            raise WeightsMissing(f"{MODEL_ID}@{MODEL_REVISION[:7]} is not in the local Hugging Face cache; "
                                 f"run setup.sh to download it ({type(e).__name__})") from e
        self.model = model.to(self.device).eval()

    def run(self, images: list, task: str, text: Optional[str] = None) -> list[dict]:
        """One batched ``generate`` call; the processor's parsed output per image."""
        torch = self.torch
        prompt = task if text is None else task + text
        inputs = self.processor(text=[prompt] * len(images), images=images, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        pixel_values = inputs["pixel_values"].to(self.device, self.dtype)
        with torch.inference_mode():
            ids = self.model.generate(input_ids=input_ids, pixel_values=pixel_values,
                                      max_new_tokens=MAX_NEW_TOKENS, num_beams=1, do_sample=False)
        texts = self.processor.batch_decode(ids, skip_special_tokens=False)
        return [self.processor.post_process_generation(t, task=task, image_size=im.size)[task]
                for t, im in zip(texts, images)]


def _is_cuda_oom(e: BaseException) -> bool:
    return type(e).__name__ == "OutOfMemoryError" or "CUDA out of memory" in str(e)


def _empty_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _get_runner() -> Optional[_Florence]:
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
            _runner = _Florence()
        except Exception as e:  # noqa: BLE001 - missing package, missing weights, CUDA trouble
            if _is_cuda_oom(e):
                _state = "cold"
                log.warning("Florence-2 did not fit on the GPU (%s); no lettering or named parts for this job",
                            type(e).__name__)
                _empty_cache()
                return None
            _state = "unavailable"
            _weights_missing = isinstance(e, WeightsMissing)
            _unavailable_reason = f"{type(e).__name__}: {e}"
            if before != "unavailable":
                log.warning("Florence-2 unavailable (%s); the regions stage runs without lettering and named parts",
                            _unavailable_reason)
            return None
        _weights_missing = False
        _state = "ready"
        log.info("Florence-2 loaded in %.1f s", time.perf_counter() - t0)
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


# ---------------------------------------------------------------------- queries

def _tiles(w: int, h: int) -> list[tuple[int, int, int, int]]:
    tw, th = int(round(TILE_FRAC * w)), int(round(TILE_FRAC * h))
    return [(0, 0, tw, th), (w - tw, 0, w, th), (0, h - th, tw, h), (w - tw, h - th, w, h)]


def _ocr(runner: _Florence, image_u8: np.ndarray) -> list[dict]:
    from PIL import Image
    h, w = image_u8.shape[:2]
    pil = Image.fromarray(image_u8)
    boxes = [(0, 0, w, h)] + (_tiles(w, h) if min(w, h) >= 64 else [])
    ims = [pil if b == (0, 0, w, h) else pil.crop(b) for b in boxes]
    outs = runner.run(ims, "<OCR_WITH_REGION>")
    res = []
    for k, (b, o) in enumerate(zip(boxes, outs)):
        for q, lab in zip(o.get("quad_boxes", []), o.get("labels", [])):
            qq = [float(q[i]) + (b[0] if i % 2 == 0 else b[1]) for i in range(8)]
            res.append({"quad": qq, "text": str(lab).replace("</s>", "").replace("<s>", ""),
                        "src": "full" if k == 0 else f"tile{k}"})
    return res


def _grounding(runner: _Florence, image_u8: np.ndarray) -> list[dict]:
    from PIL import Image
    o = runner.run([Image.fromarray(image_u8)], "<CAPTION_TO_PHRASE_GROUNDING>", CAPTION)[0]
    return [{"box": [float(v) for v in b], "label": str(lbl)} for b, lbl in zip(o.get("bboxes", []), o.get("labels", []))]


def caption(image_rgb_u8: np.ndarray) -> Optional[str]:
    """Florence-2's ``<CAPTION>`` of a work image ("A red motorcycle is parked in a studio"),
    which picks the part vocabulary of :func:`smallparts.find_kind_parts`; None when the model
    is unavailable, does not fit on the shared card or fails on this image (logged; the part
    step then uses the generic vocabulary)."""
    if image_rgb_u8.dtype != np.uint8 or image_rgb_u8.ndim != 3 or image_rgb_u8.shape[2] != 3:
        raise ValueError("caption expects an HxWx3 uint8 RGB image")
    runner = _get_runner()
    if runner is None:
        return None
    try:
        from PIL import Image
        out = runner.run([Image.fromarray(np.ascontiguousarray(image_rgb_u8))], "<CAPTION>")[0]
    except Exception as e:  # noqa: BLE001 - an enhancement, never a failure of the analysis
        if _is_cuda_oom(e):
            log.warning("Florence-2 ran out of GPU memory; no caption for this job")
            release()
        else:
            log.warning("Florence-2 caption failed on this image (%s: %s)", type(e).__name__, e)
            _empty_cache()
        return None
    return str(out).replace("</s>", "").replace("<s>", "").strip()


def ground(image_rgb_u8: np.ndarray, text: str) -> Optional[list[dict]]:
    """Boxes of ``text`` ("spring") on a work image, Florence-2's ``<CAPTION_TO_PHRASE_GROUNDING>``:
    ``[{"box": [x0, y0, x1, y1], "phrase", "score": None, "det": "florence"}]`` (Florence-2 gives no
    confidence), or None when the model is unavailable, does not fit on the shared card or fails
    (logged). Find part's fallback when OWLv2 is not available."""
    if image_rgb_u8.dtype != np.uint8 or image_rgb_u8.ndim != 3 or image_rgb_u8.shape[2] != 3:
        raise ValueError("ground expects an HxWx3 uint8 RGB image")
    runner = _get_runner()
    if runner is None:
        return None
    try:
        from PIL import Image
        o = runner.run([Image.fromarray(np.ascontiguousarray(image_rgb_u8))], "<CAPTION_TO_PHRASE_GROUNDING>",
                       str(text).strip())[0]
    except Exception as e:  # noqa: BLE001 - Find part reports "nothing found" instead
        if _is_cuda_oom(e):
            log.warning("Florence-2 ran out of GPU memory; no grounding for %r", text)
            release()
        else:
            log.warning("Florence-2 grounding failed (%s: %s)", type(e).__name__, e)
            _empty_cache()
        return None
    return [{"box": [float(v) for v in b], "phrase": str(lbl).strip() or str(text), "score": None, "det": "florence"}
            for b, lbl in zip(o.get("bboxes", []), o.get("labels", []))]


def analyse(image_rgb_u8: np.ndarray) -> Optional[dict[str, Any]]:
    """Florence-2's answers for one work image, or None when the model is unavailable, does
    not fit on the shared card or fails on this image (all logged; the stage then runs
    without it). Returns ``{"ocr": [{"quad": [8 floats], "text", "src"}], "grounding":
    [{"box": [x0, y0, x1, y1], "label"}], "seconds": {"ocr", "grounding"}}``."""
    if image_rgb_u8.dtype != np.uint8 or image_rgb_u8.ndim != 3 or image_rgb_u8.shape[2] != 3:
        raise ValueError("analyse expects an HxWx3 uint8 RGB image")
    runner = _get_runner()
    if runner is None:
        return None
    try:
        t0 = time.perf_counter()
        ocr = _ocr(runner, np.ascontiguousarray(image_rgb_u8))
        t1 = time.perf_counter()
        grounding = _grounding(runner, np.ascontiguousarray(image_rgb_u8))
        t2 = time.perf_counter()
    except Exception as e:  # noqa: BLE001 - an enhancement, never a failure of the analysis
        if _is_cuda_oom(e):
            log.warning("Florence-2 ran out of GPU memory; no lettering or named parts for this job")
            release()
        else:
            log.warning("Florence-2 failed on this image (%s: %s); no lettering or named parts for this job",
                        type(e).__name__, e)
            _empty_cache()
        return None
    return {"ocr": ocr, "grounding": grounding, "seconds": {"ocr": t1 - t0, "grounding": t2 - t1}}
