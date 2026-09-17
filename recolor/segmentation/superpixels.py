"""SLIC superpixels used to label whatever SAM left uncovered."""
from __future__ import annotations

import numpy as np

# Above this many pixels SLIC runs on a downscaled copy; superpixels are hundreds of
# pixels each, so the nearest-neighbour upsample costs nothing visible and saves ~2 s.
_SLIC_MAX_PIXELS = 900_000


def slic_labels(image_rgb_u8: np.ndarray, n_segments: int, compactness: float = 10.0) -> np.ndarray:
    """SLIC superpixels of an RGB uint8 image.

    Returns an int32 HxW label map whose values are exactly 0..K-1 (contiguous, every
    pixel assigned, K <= n_segments plus a few enforced-connectivity fragments). The
    clustering is done in CIE Lab. Deterministic for a given input.
    """
    from skimage.segmentation import slic

    from ..imageio import resize_to

    if image_rgb_u8.ndim != 3 or image_rgb_u8.shape[2] != 3:
        raise ValueError("slic_labels expects an HxWx3 RGB image")
    h, w = image_rgb_u8.shape[:2]
    n = max(1, int(n_segments))
    src = image_rgb_u8
    if h * w > _SLIC_MAX_PIXELS:
        s = (_SLIC_MAX_PIXELS / float(h * w)) ** 0.5
        src = resize_to(image_rgb_u8, (max(1, int(w * s)), max(1, int(h * s))))
    seg = slic(src, n_segments=n, compactness=float(compactness), start_label=0,
               channel_axis=-1, enforce_connectivity=True)
    seg = np.asarray(seg)
    if seg.shape != (h, w):
        seg = resize_to(seg.astype(np.int32), (w, h), nearest=True)
    _, inv = np.unique(seg, return_inverse=True)
    return inv.reshape(h, w).astype(np.int32)
