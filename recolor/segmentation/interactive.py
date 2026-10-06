"""Interactive segmentation: SAM 2.1 prompted by the user's clicks and boxes (Select part).

The studio sends the clicks (positive and negative points, in click order, in work-resolution
pixels) and an optional box. The prompt is replayed the way SAM's own demo refines a mask, but
without any state on the server: the first step is the box with the first positive point (SAM's
three candidates when ``multimask``; ``pick`` chooses the one the chain continues from, by
default the best score, for a box the best score x box agreement), and every other point is one
more step fed the previous step's low-res logits as SAM's mask input. Measured on the Ducati's
far-side front caliper (seen between the spokes of its wheel): two positive clicks in one
single-output call answered with the whole wheel (85k px); replayed they give the caliper (1.4k
px, 2.2k px with a third click). The same payload gives the same mask, so a commit re-runs it on
the server and never takes a mask from the client.

A small part is then refined on a crop: SAM's 256 px low-res logits are 6 px wide at the 1536 px
working resolution, so the chain is replayed on a window around the mask (its box grown by
CROP_PAD, at least CROP_MIN_SIDE, snapped outward to CROP_GRID so the next click usually reuses
the window's embedding) and the refined mask is taken when the window does not clip it and it
overlaps the full-image answer by REFINE_MIN_IOU. The window goes back with the answer, and a
commit that sends it back replays exactly the same steps.

Nothing here imports torch or SAM: :class:`~recolor.segmentation.sam_masks.PromptSession` runs the
model, and any object with its ``predict`` and ``shape`` does for tests.
"""
from __future__ import annotations

import base64
import io
import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import cv2
import numpy as np

from .userparts import MIN_PIECE_PX, drop_islands

#: At most this many points in one prompt.
MAX_POINTS = 40
#: A point may lie this many px outside the image (a click on the last pixel column); it is clamped.
POINT_SLACK = 1.0
#: Masks whose extent is at most this share of the image's long side are refined on a crop.
REFINE_MAX_SIDE = 0.45
#: The crop is the mask's extent (with the points and the box) grown by this share of its size ...
CROP_PAD = 0.35
#: ... and at least this many px on each side ...
CROP_PAD_MIN = 16
#: ... at least this big on each side, at most this elongated ...
CROP_MIN_SIDE = 160
CROP_MAX_ASPECT = 2.0
#: ... with its edges snapped outward to this grid (consecutive clicks on one part share a crop).
CROP_GRID = 16
#: A refined mask must overlap the full-image answer by this IoU to replace it.
REFINE_MIN_IOU = 0.5


class PromptError(ValueError):
    """A prompt the model cannot be asked (bad coordinates, no positive point, ...)."""


@dataclass(frozen=True)
class Prompt:
    """A validated prompt in work-resolution pixels: ``points`` ``(x, y, 1 | 0)`` in click order,
    an optional ``box`` ``(x0, y0, x1, y1)``, SAM's ``multimask`` for the first step, the first
    step's candidate ``pick`` (None: the best), and a refinement window ``crop`` a previous answer
    gave (None: chosen from the mask)."""
    points: tuple[tuple[float, float, int], ...] = ()
    box: Optional[tuple[float, float, float, float]] = None
    multimask: bool = True
    pick: Optional[int] = None
    crop: Optional[tuple[int, int, int, int]] = None

    @property
    def empty(self) -> bool:
        return not self.points and self.box is None

    def ordered_points(self) -> list[tuple[float, float, int]]:
        """The points in replay order: the first positive one first, the rest in click order."""
        pts = list(self.points)
        first = next((i for i, p in enumerate(pts) if p[2] == 1), None)
        if first is None:
            return pts
        return [pts[first]] + pts[:first] + pts[first + 1:]

    @property
    def steps(self) -> int:
        return 0 if self.empty else max(1, len(self.points))

    def to_json(self) -> dict:
        return {"points": [[round(x, 2), round(y, 2), int(l)] for x, y, l in self.points],
                "box": None if self.box is None else [round(v, 2) for v in self.box],
                "multimask": bool(self.multimask), "pick": self.pick,
                "crop": None if self.crop is None else [int(v) for v in self.crop]}


def _num(v: Any, what: str) -> float:
    """``v`` as a finite float, else PromptError: a bool, a string, None, NaN, an infinity, and an
    integer too large for a float (JSON numbers may have any number of digits: a 401-digit x
    raised OverflowError, a 500)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise PromptError(f"{what} must be a finite number")
    try:
        f = float(v)
    except (OverflowError, ValueError):
        raise PromptError(f"{what} must be a finite number") from None
    if not math.isfinite(f):
        raise PromptError(f"{what} must be a finite number")
    return f


def _px(v: float) -> str:
    """A coordinate for an error message (a huge one in short form, not 300 digits)."""
    return f"{v:.0f}" if abs(v) < 1e7 else f"{v:.3g}"


def parse_prompt(body: Any, width: int, height: int) -> Prompt:
    """Validate a prompt body ``{points: [[x, y, 1|0], ...], box: [x0, y0, x1, y1] | null,
    multimask: bool, pick?: 0-2 | null, crop?: [x0, y0, x1, y1] | null}`` against a ``width`` x
    ``height`` work image. Points may lie POINT_SLACK px outside the image (clamped); a box is
    clipped to the image. Raises :class:`PromptError` with a message for the user. An empty
    prompt (no points, no box) is valid: it only prepares the embedding."""
    if not isinstance(body, dict):
        raise PromptError("the prompt must be a JSON object")
    raw = body.get("points")
    raw = [] if raw is None else raw
    if not isinstance(raw, (list, tuple)):
        raise PromptError("points must be a list of [x, y, 1 | 0]")
    if len(raw) > MAX_POINTS:
        raise PromptError(f"at most {MAX_POINTS} points per prompt")
    points: list[tuple[float, float, int]] = []
    for p in raw:
        if not isinstance(p, (list, tuple)) or len(p) not in (2, 3):
            raise PromptError("each point must be [x, y, 1 | 0]")
        x, y = _num(p[0], "a point's x"), _num(p[1], "a point's y")
        lab = 1 if len(p) == 2 else p[2]
        if isinstance(lab, bool):
            lab = int(lab)
        # an integer: 1.0 is no label (the studio sends 1 and 0)
        if not isinstance(lab, int) or lab not in (0, 1):
            raise PromptError("a point's label must be 1 (part) or 0 (not the part)")
        if not (-POINT_SLACK <= x <= width - 1 + POINT_SLACK and -POINT_SLACK <= y <= height - 1 + POINT_SLACK):
            raise PromptError(f"point ({_px(x)}, {_px(y)}) lies outside the {width} x {height} work image")
        points.append((min(max(x, 0.0), width - 1.0), min(max(y, 0.0), height - 1.0), int(lab)))
    box = body.get("box")
    bx: Optional[tuple[float, float, float, float]] = None
    if box is not None:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise PromptError("box must be [x0, y0, x1, y1] or null")
        x0, y0, x1, y1 = (_num(v, "a box coordinate") for v in box)
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(float(width), x1), min(float(height), y1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            raise PromptError("the box must cover at least 2 x 2 px of the image")
        bx = (x0, y0, x1, y1)
    multimask = body.get("multimask", True)
    if not isinstance(multimask, bool):
        raise PromptError("multimask must be true or false")
    pick = body.get("pick")
    if pick is not None:
        if isinstance(pick, bool) or not isinstance(pick, int) or not (0 <= pick <= 2):
            raise PromptError("pick must be 0, 1, 2 or null")
    crop = body.get("crop")
    cr: Optional[tuple[int, int, int, int]] = None
    if crop is not None:
        if not isinstance(crop, (list, tuple)) or len(crop) != 4:
            raise PromptError("crop must be [x0, y0, x1, y1] or null")
        c = [int(round(_num(v, "a crop coordinate"))) for v in crop]
        if 0 <= c[0] < c[2] <= width and 0 <= c[1] < c[3] <= height and c[2] - c[0] >= 16 and c[3] - c[1] >= 16:
            cr = (c[0], c[1], c[2], c[3])
    if points and bx is None and not any(p[2] == 1 for p in points):
        raise PromptError("add a positive point (a click on the part) or draw a box")
    return Prompt(tuple(points), bx, multimask, pick, cr)


# ---------------------------------------------------------------------- masks

def fill_small_holes(mask: np.ndarray, min_px: int = MIN_PIECE_PX) -> np.ndarray:
    """``mask`` with its 4-connected holes under ``min_px`` filled (a hole touching the image
    border is no hole). Works on the mask's box grown by 1 px, where a background piece touching
    the window's edge is the outside (one full-image pass per hole cost a one-click prompt 50 ms)."""
    m = np.asarray(mask, bool)
    if not m.any() or min_px <= 1:
        return m.copy()
    H, W = m.shape
    rows, cols = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
    y0, y1 = max(0, int(rows[0]) - 1), min(H, int(rows[-1]) + 2)
    x0, x1 = max(0, int(cols[0]) - 1), min(W, int(cols[-1]) + 2)
    win = m[y0:y1, x0:x1]
    k, cc, st, _ = cv2.connectedComponentsWithStats((~win).astype(np.uint8), connectivity=4)
    out = m.copy()
    if k <= 1:
        return out
    hh, ww = win.shape
    x, y, w, h, a = (st[1:, i] for i in range(5))
    ids = np.flatnonzero((a < min_px) & (x > 0) & (y > 0) & (x + w < ww) & (y + h < hh)) + 1
    if ids.size:
        out[y0:y1, x0:x1] |= np.isin(cc, ids)
    return out


def clean(mask: np.ndarray, min_px: int = MIN_PIECE_PX) -> np.ndarray:
    """What becomes of a SAM mask: pieces under ``min_px`` dropped, holes under it filled
    (the carve then keeps the real holes, :func:`userparts.carve`)."""
    return fill_small_holes(drop_islands(mask, min_px), min_px)


def _bbox(m: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    ys, xs = np.nonzero(m)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = int((a | b).sum())
    return float((a & b).sum()) / u if u else 0.0


def _box_iou(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-6)


def _clipped(m: np.ndarray, crop: tuple[int, int, int, int], width: int, height: int) -> bool:
    """True when mask ``m`` (in the window ``crop``) touches a window side that is not the image's."""
    x0, y0, x1, y1 = crop
    return (y0 > 0 and bool(m[0].any())) or (y1 < height and bool(m[-1].any())) \
        or (x0 > 0 and bool(m[:, 0].any())) or (x1 < width and bool(m[:, -1].any()))


def choose_crop(mask: np.ndarray, prompt: Prompt, width: int, height: int) -> Optional[tuple[int, int, int, int]]:
    """The refinement window of a mask (see the module docstring), or None when the part is too
    large to need one (its extent above REFINE_MAX_SIDE of the image's long side) or the window
    would be the whole image."""
    ext = []
    bb = _bbox(mask)
    if bb is not None:
        ext.append(bb)
    for x, y, lab in prompt.points:
        if lab == 1:
            ext.append((x, y, x + 1, y + 1))
    if prompt.box is not None:
        ext.append(prompt.box)
    if not ext:
        return None
    x0, y0 = min(e[0] for e in ext), min(e[1] for e in ext)
    x1, y1 = max(e[2] for e in ext), max(e[3] for e in ext)
    size = max(x1 - x0, y1 - y0)
    if size > REFINE_MAX_SIDE * max(width, height):
        return None
    pad = max(CROP_PAD_MIN, CROP_PAD * size)
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    w, h = x1 - x0, y1 - y0
    side = max(CROP_MIN_SIDE, max(w, h) / CROP_MAX_ASPECT)
    if w < side:
        x0, x1 = x0 - (side - w) / 2, x1 + (side - w) / 2
    if h < side:
        y0, y1 = y0 - (side - h) / 2, y1 + (side - h) / 2
    g = CROP_GRID
    c = (max(0, int(math.floor(x0 / g)) * g), max(0, int(math.floor(y0 / g)) * g),
         min(width, int(math.ceil(x1 / g)) * g), min(height, int(math.ceil(y1 / g)) * g))
    if c[2] - c[0] < 16 or c[3] - c[1] < 16 or (c[0] == 0 and c[1] == 0 and c[2] == width and c[3] == height):
        return None
    return c


def _usable_crop(crop: Optional[tuple[int, int, int, int]], prompt: Prompt) -> Optional[tuple[int, int, int, int]]:
    """A window from the client, when it still holds every positive point and the box."""
    if crop is None:
        return None
    x0, y0, x1, y1 = crop
    for x, y, lab in prompt.points:
        if lab == 1 and not (x0 <= x < x1 and y0 <= y < y1):
            return None
    if prompt.box is not None:
        b = prompt.box
        if b[0] < x0 or b[1] < y0 or b[2] > x1 or b[3] > y1:
            return None
    return crop


# ---------------------------------------------------------------------- the replay

@dataclass
class Chain:
    """One replay: the first step's candidates ``[(mask, score)]``, the one ``pick`` continued
    from, the final ``mask`` and ``score`` (in the frame the chain ran in), how many ``steps`` it
    took and whether the window's embedding was ``computed`` for it."""
    first: list[tuple[np.ndarray, float]]
    pick: int
    mask: np.ndarray
    score: float
    steps: int
    computed: bool


def _arrays(points: Sequence[tuple[float, float, int]], ox: float, oy: float):
    if not points:
        return None, None
    return (np.array([[x - ox, y - oy] for x, y, _ in points], np.float32),
            np.array([int(l) for *_, l in points], np.int32))


def run_chain(session: Any, prompt: Prompt, crop: Optional[tuple[int, int, int, int]] = None,
              match: Optional[np.ndarray] = None) -> Chain:
    """Replay ``prompt`` on ``session`` (the full image, or the window ``crop``): the box with the
    first positive point first, then one point per step with the previous step's low-res logits
    as the mask input. The first step's candidate is ``match``'s best overlap when given (a full-
    image mask: the refinement continues from the candidate the full-image answer continued
    from), else ``prompt.pick``, else the best score (x box agreement for a box)."""
    ox, oy = (crop[0], crop[1]) if crop is not None else (0, 0)
    pts = prompt.ordered_points()
    if crop is not None:
        x0, y0, x1, y1 = crop
        pts = [p for p in pts if p[2] == 1 or (x0 <= p[0] < x1 and y0 <= p[1] < y1)]
    box = None
    if prompt.box is not None:
        b = prompt.box
        if crop is not None:
            b = (max(b[0], crop[0]), max(b[1], crop[1]), min(b[2], crop[2]), min(b[3], crop[3]))
        box = np.array([b[0] - ox, b[1] - oy, b[2] - ox, b[3] - oy], np.float32)
    pc, pl = _arrays(pts[:1], ox, oy)
    masks, scores, low, computed = session.predict(pc, pl, box, None, prompt.multimask, crop=crop)
    first = [(np.asarray(masks[i], bool), float(scores[i])) for i in range(len(scores))]
    if not first:
        raise RuntimeError("SAM returned no mask")
    if match is not None:
        win = match if crop is None else match[crop[1]:crop[3], crop[0]:crop[2]]
        H, W = session.shape
        keys = [(not (crop is not None and _clipped(m, crop, W, H)), _iou(m, win), s) for m, s in first]
        k = max(range(len(first)), key=lambda i: keys[i])
    elif prompt.pick is not None and prompt.pick < len(first):
        k = int(prompt.pick)
    elif box is not None:
        fit = []
        for m, s in first:
            bb = _bbox(m)
            fit.append(s * (_box_iou(bb, box) if bb is not None else 0.0))
        k = int(np.argmax(fit))
    else:
        k = int(np.argmax([s for _, s in first]))
    mask, score = first[k]
    logits = low[k][None]
    for i in range(2, len(pts) + 1):
        pc, pl = _arrays(pts[:i], ox, oy)
        masks, scores, low, _ = session.predict(pc, pl, box, logits, False, crop=crop)
        mask, score = np.asarray(masks[0], bool), float(scores[0])
        logits = low[0][None]
    return Chain(first, k, mask, score, max(1, len(pts)), computed)


# ---------------------------------------------------------------------- the answer

@dataclass
class Candidate:
    """A full-size mask with SAM's score; ``index`` is its place among the first step's
    candidates, ``refined`` whether it comes from the crop."""
    mask: np.ndarray
    score: float
    index: int
    refined: bool = False

    @property
    def area(self) -> int:
        return int(self.mask.sum())


@dataclass
class Answer:
    """What a prompt gave: the ``mask`` to show and commit, the first step's other candidates
    (``alternatives``, for a one-step prompt with multimask), the ``pick`` the chain continued
    from, the refinement ``crop`` it tried (None: none), the ``steps`` and ``timings`` (ms)."""
    mask: Candidate
    alternatives: list[Candidate]
    pick: int
    crop: Optional[tuple[int, int, int, int]]
    steps: int
    timings: dict = field(default_factory=dict)


def segment(session: Any, prompt: Prompt, refine: bool = True) -> Answer:
    """Replay ``prompt`` on the full image and, for a small part, on its refinement window (see
    the module docstring). Guarantees: the same ``prompt`` (with the answer's ``crop`` sent back)
    gives the same mask; every mask is full-size bool, with pieces and holes under MIN_PIECE_PX
    cleaned (:func:`clean`)."""
    if prompt.empty:
        raise PromptError("click on the part or draw a box around it")
    H, W = session.shape
    t0 = time.perf_counter()
    full = run_chain(session, prompt)
    t1 = time.perf_counter()
    mask = clean(full.mask)
    chosen = Candidate(mask, full.score, full.pick)
    alternatives = [Candidate(clean(m), s, i) for i, (m, s) in enumerate(full.first) if i != full.pick] \
        if full.steps == 1 and len(full.first) > 1 else []
    crop = None
    crop_computed = False
    t2 = t1
    if refine and mask.any():
        crop = _usable_crop(prompt.crop, prompt) or choose_crop(mask, prompt, W, H)
        if crop is not None:
            ref = run_chain(session, prompt, crop=crop, match=full.first[full.pick][0])
            crop_computed = ref.computed
            x0, y0, x1, y1 = crop
            m = np.zeros((H, W), bool)
            m[y0:y1, x0:x1] = ref.mask
            if not _clipped(ref.mask, crop, W, H) and _iou(m, mask) >= REFINE_MIN_IOU:
                chosen = Candidate(clean(m), ref.score, full.pick, refined=True)
            t2 = time.perf_counter()
    timings = {"chain_ms": round((t1 - t0) * 1000, 1), "refine_ms": round((t2 - t1) * 1000, 1),
               "crop_embed": "computed" if crop_computed else ("cached" if crop is not None else None)}
    return Answer(chosen, alternatives, full.pick, crop, full.steps, timings)


def encode_mask(mask: np.ndarray) -> dict:
    """A mask for the client: ``{"png": a 1-bit PNG data URL of the mask's bounding box, "bbox":
    [x0, y0, x1, y1] (exclusive), "area"}`` (``png`` and ``bbox`` None for an empty mask)."""
    bb = _bbox(mask)
    if bb is None:
        return {"png": None, "bbox": None, "area": 0}
    from PIL import Image
    x0, y0, x1, y1 = bb
    crop = np.ascontiguousarray(mask[y0:y1, x0:x1])
    buf = io.BytesIO()
    Image.fromarray(crop.astype(np.uint8) * 255).convert("1").save(buf, format="PNG")
    return {"png": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii"),
            "bbox": [x0, y0, x1, y1], "area": int(crop.sum())}


def answer_json(answer: Answer, width: int, height: int) -> dict:
    """The JSON of an answer (``POST /api/jobs/{id}/segment``)."""
    total = float(max(1, width * height))

    def cand(c: Candidate) -> dict:
        d = encode_mask(c.mask)
        d.update(score=round(float(c.score), 4), index=int(c.index), refined=bool(c.refined),
                 area_frac=round(d["area"] / total, 6))
        return d

    return {"mask": cand(answer.mask), "alternatives": [cand(c) for c in answer.alternatives],
            "pick": int(answer.pick), "crop": None if answer.crop is None else [int(v) for v in answer.crop],
            "steps": int(answer.steps), "size": [int(width), int(height)], "timings": dict(answer.timings)}
