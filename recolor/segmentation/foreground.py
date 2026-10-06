"""Foreground matte at analysis time: BiRefNet (Zheng et al. 2024, MIT, ``ZhengPeng7/
BiRefNet_dynamic``), the prior that tells the object from its backdrop.

The border rule alone ("the group owning most of the image border") took the Ducati's rear
wheel, tyre, swingarm and engine internals into the backdrop group, and flagged one of the
RX-78's two backdrop halves. :func:`fg_prob` gives the regions stage a soft matte (float32
HxW in [0, 1]: 1 = object) that :func:`hierarchy.cut_on_matte` cuts regions along and
:func:`grouping.backdrop_decisions` reads per region, so that every backdrop group is
flagged and no part of the object lands in one.

The model is a lazy singleton like SAM 2, the intrinsic model, ViTMatte and Florence-2
(about 0.5 s to load, 2.3 GB peak allocated for a 1536-px image, 60-160 ms per image on the
shared card), released by the pipeline's idle watchdog. The weights are read from the local
Hugging Face cache only (``setup.sh`` downloads the pinned snapshot, whose own model code is
part of it): the analysis never goes to the network. Without ``transformers`` or the weights
the shipping border rule decides the background (logged once; missing weights are looked for
again on the next job), and so it does for a job on which the model runs out of GPU memory.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np

log = logging.getLogger("recolor.segmentation.foreground")

MODEL_ID = "ZhengPeng7/BiRefNet_dynamic"
#: The Hugging Face snapshot the thresholds were measured with.
MODEL_REVISION = "280306042f57b7a33854319da62fd86aaa89ec4c"
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)

_lock = threading.Lock()
_runner: Optional["_BiRefNet"] = None
_state = "cold"                  # cold | loading | ready | unavailable
_unavailable_reason: Optional[str] = None
_weights_missing = False


class WeightsMissing(OSError):
    """The BiRefNet snapshot is not in the local Hugging Face cache (``setup.sh`` downloads it)."""


#: The snapshot's files that must be in the local cache before the model is loaded (the model
#: code, which ``trust_remote_code`` runs, is part of the pinned snapshot).
SNAPSHOT_FILES = ("config.json", "birefnet.py", "BiRefNet_config.py", "model.safetensors")


def _snapshot_cached() -> bool:
    """True when the pinned snapshot is in the local cache (:mod:`hfcache`, no network)."""
    from .hfcache import snapshot_present
    return snapshot_present(MODEL_ID, MODEL_REVISION, SNAPSHOT_FILES)


class _BiRefNet:
    """BiRefNet_dynamic on the GPU (fp16 on CUDA), loaded from the local cache only. A missing
    snapshot raises :class:`WeightsMissing` before ``from_pretrained`` is called (which asks the
    network for its error message even with ``local_files_only``)."""

    def __init__(self) -> None:
        if not _snapshot_cached():
            raise WeightsMissing(f"{MODEL_ID}@{MODEL_REVISION[:7]} is not in the local Hugging Face cache; "
                                 f"run setup.sh to download it")
        import torch
        from transformers import AutoModelForImageSegmentation
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model = AutoModelForImageSegmentation.from_pretrained(MODEL_ID, trust_remote_code=True,
                                                                  revision=MODEL_REVISION, local_files_only=True)
        except OSError as e:
            raise WeightsMissing(f"{MODEL_ID}@{MODEL_REVISION[:7]} is not in the local Hugging Face cache; "
                                 f"run setup.sh to download it ({type(e).__name__})") from e
        self.model = model.to(self.device).eval()
        self.half = self.device.type == "cuda"
        if self.half:
            self.model.half()

    def matte(self, image_u8: np.ndarray) -> np.ndarray:
        """float32 HxW in [0, 1]: the sigmoid of the last output on the image padded (edge
        replicate) to a multiple of 32, cropped back."""
        torch = self.torch
        import torch.nn.functional as F
        h, w = image_u8.shape[:2]
        H2, W2 = (h + 31) // 32 * 32, (w + 31) // 32 * 32
        x = torch.from_numpy(np.ascontiguousarray(image_u8)).to(self.device).permute(2, 0, 1)[None].float() / 255.0
        if (H2, W2) != (h, w):
            x = F.pad(x, (0, W2 - w, 0, H2 - h), mode="replicate")
        mean = torch.tensor(_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(_STD, device=self.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        if self.half:
            x = x.half()
        with torch.inference_mode():
            p = self.model(x)[-1].sigmoid()
        out = p[0, 0, :h, :w].float().cpu().numpy()
        return np.clip(out, 0.0, 1.0).astype(np.float32)


def _is_cuda_oom(e: BaseException) -> bool:
    return type(e).__name__ == "OutOfMemoryError" or "CUDA out of memory" in str(e)


def _empty_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _get_runner() -> Optional[_BiRefNet]:
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
            _runner = _BiRefNet()
        except Exception as e:  # noqa: BLE001
            if _is_cuda_oom(e):
                _state = "cold"
                log.warning("BiRefNet did not fit on the GPU (%s); border rule for the background of this job",
                            type(e).__name__)
                _empty_cache()
                return None
            _state = "unavailable"
            _weights_missing = isinstance(e, WeightsMissing)
            _unavailable_reason = f"{type(e).__name__}: {e}"
            if before != "unavailable":
                log.warning("BiRefNet unavailable (%s); the background is decided by the border rule", _unavailable_reason)
            return None
        _weights_missing = False
        _state = "ready"
        log.info("BiRefNet loaded in %.1f s", time.perf_counter() - t0)
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


def fg_prob(image_rgb_u8: np.ndarray) -> Optional[np.ndarray]:
    """The foreground matte of a work image (float32 HxW in [0, 1], 1 = the salient object),
    or None when the model is unavailable, does not fit on the shared card or fails on this
    image (all logged: the caller then falls back to the border rule)."""
    if image_rgb_u8.dtype != np.uint8 or image_rgb_u8.ndim != 3 or image_rgb_u8.shape[2] != 3:
        raise ValueError("fg_prob expects an HxWx3 uint8 RGB image")
    runner = _get_runner()
    if runner is None:
        return None
    try:
        return runner.matte(image_rgb_u8)
    except Exception as e:  # noqa: BLE001 - an enhancement, never a failure of the analysis
        if _is_cuda_oom(e):
            log.warning("BiRefNet ran out of GPU memory; border rule for the background of this job")
            release()
        else:
            log.warning("BiRefNet failed on this image (%s: %s); border rule for the background of this job",
                        type(e).__name__, e)
            _empty_cache()
        return None
